#!/usr/bin/env bash
#
# Test d'un endpoint RunPod Serverless (résumé juridique FR)
# Usage:
#  ./test_runpod_summarizer.sh -k <RUNPOD_API_KEY> -e <ENDPOINT_ID> \
#     [--text "Texte à résumer"] [--pdf-url "https://.../doc.pdf"] [--pdf "local.pdf"] \
#     [--anonymiser true|false] [--timeout 120] [--active-rate 0.00013]
#
# Par défaut, appelle /runsync (synchrone).
# Estimation de coût basée sur tarif/secondes (option --active-rate).
#
set -euo pipefail

API_KEY="rpa_REN14QO52RCZGWQ4IL3Y5AQML5VL9X1DDG27MD4416at39"
ENDPOINT_ID="upaqvz9xizruai"
TEXT=""
PDF_URL=""
PDF_FILE=""
ANON="true"
TIMEOUT="120"
# Tarif "Active" 4090 par défaut (0.00013 $/s) ; adapte à 3090/L4 = 0.00013 aussi selon ta config
# Cf. doc: 24GB (L4, A5000, 3090) $0.00013/s Active ; 4090 PRO $0.00021/s Active
# https://docs.runpod.io/serverless/endpoints/endpoint-configurations
ACTIVE_RATE="0.00013"

usage() {
  grep -E '^#' "$0" | sed -e 's/^# \{0,1\}//'
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--api-key) API_KEY="$2"; shift 2;;
    -e|--endpoint) ENDPOINT_ID="$2"; shift 2;;
    --text) TEXT="$2"; shift 2;;
    --pdf-url) PDF_URL="$2"; shift 2;;
    --pdf) PDF_FILE="$2"; shift 2;;
    --anonymiser) ANON="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --active-rate) ACTIVE_RATE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Argument inconnu: $1"; usage; exit 1;;
  esac
done

if [[ -z "$API_KEY" || -z "$ENDPOINT_ID" ]]; then
  echo "Erreur: --api-key et --endpoint sont obligatoires."
  usage; exit 2
fi

# Prépare l'input JSON
INPUT_JSON="{}"
if [[ -n "$TEXT" ]]; then
  INPUT_JSON=$(jq -n --arg t "$TEXT" --arg a "$ANON" '{input:{text:$t, anonymiser:($a=="true")}}')
elif [[ -n "$PDF_URL" ]]; then
  INPUT_JSON=$(jq -n --arg u "$PDF_URL" --arg a "$ANON" '{input:{pdf_url:$u, anonymiser:($a=="true"), ocr_fallback:true}}')
elif [[ -n "$PDF_FILE" ]]; then
  if ! command -v base64 >/dev/null; then echo "base64 requis"; exit 3; fi
  B64=$(base64 -w 0 "$PDF_FILE")
  INPUT_JSON=$(jq -n --arg b "$B64" --arg a "$ANON" '{input:{pdf_base64:$b, anonymiser:($a=="true"), ocr_fallback:true}}')
else
  echo "Erreur: fournissez --text ou --pdf-url ou --pdf"; exit 4
fi

API_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync"  # synchrone
# Docs: /runsync pour résultats immédiats; /run pour async + /status pour suivi. 
# https://docs.runpod.io/serverless/endpoints/operations

echo "==> Appel /runsync ..."
START=$(date +%s)
RESP=$(curl -s -X POST "$API_URL" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$INPUT_JSON")
END=$(date +%s)
DUR=$((END-START))

echo "Durée: ${DUR}s"
SUMMARY=$(echo "$RESP" | jq -r '.output.summary // .summary // .output // .error // empty')
if [[ -z "$SUMMARY" ]]; then
  echo "$RESP" | jq .
else
  echo
  echo "===== RÉSUMÉ ====="
  echo "$SUMMARY"
  echo "=================="
fi

# Estimation coût (approx) côté serveurless Active:
# coût ≈ DUR * ACTIVE_RATE ($/s)
COST=$(awk -v s="$DUR" -v r="$ACTIVE_RATE" 'BEGIN{printf "%.4f", s*r}')
echo "Estimation coût (Active @$${ACTIVE_RATE}/s): ~$$${COST}"

