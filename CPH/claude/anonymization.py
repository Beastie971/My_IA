#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Système d'anonymisation pour OCR Juridique v7
"""

import re

# =============================================================================
# SYSTÈME D'ANONYMISATION
# =============================================================================

class AnonymizationManager:
    """Gestionnaire d'anonymisation avec références cohérentes."""
    
    def __init__(self):
        self.person_counter = 0
        self.company_counter = 0
        self.place_counter = 0
        self.replacements = {}
    
    def get_unique_reference(self, entity_type: str, original: str) -> str:
        """Génère une référence unique pour une entité."""
        if original in self.replacements:
            return self.replacements[original]
        
        if entity_type == "person":
            self.person_counter += 1
            ref = f"[Personne-{self.person_counter}]"
        elif entity_type == "company":
            self.company_counter += 1
            ref = f"[Société-{self.company_counter}]"
        elif entity_type == "place":
            self.place_counter += 1
            ref = f"[Lieu-{self.place_counter}]"
        else:
            ref = "[Entité]"
        
        self.replacements[original] = ref
        return ref
    
    def get_mapping_report(self) -> str:
        """Génère un rapport des anonymisations effectuées."""
        if not self.replacements:
            return "Aucune anonymisation effectuée."
        
        report = ["=== RAPPORT D'ANONYMISATION ===\n"]
        
        persons = {k: v for k, v in self.replacements.items() if "[Personne-" in v}
        companies = {k: v for k, v in self.replacements.items() if "[Société-" in v}
        places = {k: v for k, v in self.replacements.items() if "[Lieu-" in v}
        
        if persons:
            report.append("PERSONNES :")
            for original, replacement in sorted(persons.items()):
                report.append(f"  {original} → {replacement}")
            report.append("")
        
        if companies:
            report.append("SOCIÉTÉS :")
            for original, replacement in sorted(companies.items()):
                report.append(f"  {original} → {replacement}")
            report.append("")
        
        if places:
            report.append("LIEUX :")
            for original, replacement in sorted(places.items()):
                report.append(f"  {original} → {replacement}")
            report.append("")
        
        report.append(f"Total : {len(self.replacements)} entité(s) anonymisée(s)")
        return "\n".join(report)

def anonymize_text(text: str) -> tuple:
    """Anonymise un texte en remplaçant les données personnelles."""
    if not text or not text.strip():
        return text, ""
    
    anonymizer = AnonymizationManager()
    result = text
    
    # Prénoms français courants
    french_firstnames = [
        "Pierre", "Jean", "Michel", "André", "Philippe", "Alain", "Bernard", "Claude", "Daniel", "Jacques",
        "François", "Henri", "Louis", "Marcel", "Paul", "Robert", "Roger", "Serge", "Christian", "Gérard",
        "Maurice", "Raymond", "René", "Guy", "Antoine", "Julien", "Nicolas", "Olivier", "Pascal", "Patrick",
        "Stéphane", "Thierry", "Vincent", "Xavier", "Yves", "Alexandre", "Christophe", "David", "Frédéric",
        "Laurent", "Sébastien", "Éric", "Fabrice", "Guillaume", "Jérôme", "Ludovic", "Mathieu", "Maxime",
        "Thomas", "Adrien", "Arthur", "Hugo", "Lucas", "Nathan", "Raphaël", "Gabriel", "Léo", "Adam",
        "Marie", "Monique", "Françoise", "Isabelle", "Catherine", "Sylvie", "Anne", "Christine", "Martine",
        "Brigitte", "Jacqueline", "Nathalie", "Chantal", "Nicole", "Véronique", "Dominique", "Christiane",
        "Patricia", "Céline", "Corinne", "Sandrine", "Valérie", "Karine", "Stéphanie", "Sophie", "Laurence",
        "Julie", "Carole", "Caroline", "Élisabeth", "Hélène", "Agnès", "Pascale", "Mireille", "Danielle",
        "Sylviane", "Florence", "Virginie", "Aurélie", "Émilie", "Mélanie", "Sarah", "Amélie", "Claire",
        "Charlotte", "Léa", "Manon", "Emma", "Chloé", "Camille", "Océane", "Marie-Christine", "Anne-Marie"
    ]
    
    # Noms de famille français courants
    french_surnames = [
        "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand", "Leroy", "Moreau",
        "Simon", "Laurent", "Lefebvre", "Michel", "Garcia", "David", "Bertrand", "Roux", "Vincent", "Fournier",
        "Morel", "Girard", "André", "Lefèvre", "Mercier", "Dupont", "Lambert", "Bonnet", "François", "Martinez",
        "Legrand", "Garnier", "Faure", "Rousseau", "Blanc", "Guerin", "Muller", "Henry", "Roussel", "Nicolas",
        "Perrin", "Morin", "Mathieu", "Clement", "Gauthier", "Dumont", "Lopez", "Fontaine", "Chevalier", "Robin",
        "Masson", "Sanchez", "Gerard", "Nguyen", "Boyer", "Denis", "Lemaire", "Duval", "Gautier", "Hernandez"
    ]
    
    # Patterns de civilité
    civility_patterns = [
        r"\b(?:M\.|Mr\.|Monsieur)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b",
        r"\b(?:Mme|Madame)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b",
        r"\b(?:Mlle|Mademoiselle)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b",
        r"\b(?:Dr|Docteur)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b",
        r"\b(?:Me|Maître)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b"
    ]
    
    for pattern in civility_patterns:
        matches = re.finditer(pattern, result, re.IGNORECASE)
        for match in reversed(list(matches)):
            full_name = match.group(1)
            title = result[match.start():match.start(1)].strip()
            replacement = anonymizer.get_unique_reference("person", full_name)
            result = result[:match.start()] + title + " " + replacement + result[match.end():]
    
    # Anonymisation des prénoms avec contexte
    for firstname in french_firstnames:
        pattern = r'\b(' + re.escape(firstname) + r')\b'
        matches = list(re.finditer(pattern, result, re.IGNORECASE))
        for match in reversed(matches):
            original = match.group(1)
            start_pos = max(0, match.start() - 20)
            end_pos = min(len(result), match.end() + 20)
            context = result[start_pos:end_pos].lower()
            
            if any(indicator in context for indicator in [
                'monsieur', 'madame', 'mademoiselle', 'appelant', 'défendeur', 
                'demandeur', 'salarié', 'employé', 'directeur', 'gérant'
            ]):
                replacement = anonymizer.get_unique_reference("person", original)
                result = result[:match.start()] + replacement + result[match.end():]
    
    # Noms de famille
    for surname in french_surnames:
        pattern = r'\b(' + re.escape(surname) + r')\b'
        matches = list(re.finditer(pattern, result))
        for match in reversed(matches):
            if match.group(1)[0].isupper():
                original = match.group(1)
                replacement = anonymizer.get_unique_reference("person", original)
                result = result[:match.start()] + replacement + result[match.end():]
    
    # Patterns de sociétés
    company_patterns = [
        r'\b([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\s+(?:SARL|SAS|SA|EURL|SNC|SCI|SASU)\b',
        r'\b(?:SARL|SAS|SA|EURL|SNC|SCI|SASU)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b',
        r'\b(?:Société|Entreprise|Établissements?)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][A-Za-zàâäçéèêëîïôöùûüÿñ\s]+)\b',
        r'\b([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][A-Za-zàâäçéèêëîïôöùûüÿñ\s&]+)\s+(?:et\s+(?:Fils|Associés|Cie))\b'
    ]
    
    for pattern in company_patterns:
        matches = re.finditer(pattern, result, re.IGNORECASE)
        for match in reversed(list(matches)):
            company_name = match.group(1).strip()
            if len(company_name) > 2:
                replacement = anonymizer.get_unique_reference("company", company_name)
                result = result.replace(match.group(0), match.group(0).replace(company_name, replacement))
    
    # Adresses
    address_patterns = [
        r'\b\d+,?\s+(?:rue|avenue|boulevard|place|impasse|allée|chemin|route)\s+([A-Za-zàâäçéèêëîïôöùûüÿñ\s\-\']+)\b',
        r'\b(?:rue|avenue|boulevard|place|impasse|allée|chemin|route)\s+([A-Za-zàâäçéèêëîïôöùûüÿñ\s\-\']+)\b'
    ]
    
    for pattern in address_patterns:
        matches = re.finditer(pattern, result, re.IGNORECASE)
        for match in reversed(list(matches)):
            street_name = match.group(1).strip()
            if len(street_name) > 3:
                replacement = anonymizer.get_unique_reference("place", street_name)
                result = result.replace(match.group(0), match.group(0).replace(street_name, replacement))
    
    # Codes postaux et villes
    postal_pattern = r'\b(\d{5})\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ\s\-]+)\b'
    matches = re.finditer(postal_pattern, result)
    for match in reversed(list(matches)):
        city = match.group(2).strip()
        if len(city) > 2:
            city_replacement = anonymizer.get_unique_reference("place", city)
            postal_replacement = "[Code-Postal]"
            result = result[:match.start()] + postal_replacement + " " + city_replacement + result[match.end():]
    
    # Téléphones, emails, montants
    result = re.sub(r'\b0[1-9](?:[.\-\s]?\d{2}){4}\b', '[Téléphone]', result)
    result = re.sub(r'\b(?:\+33\s?[1-9]|0[1-9])(?:[.\-\s]?\d{2}){4}\b', '[Téléphone]', result)
    result = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[Email]', result)
    result = re.sub(r'\b\d+(?:\.\d{3})*,\d{2}\s*(?:euros?|€)\b', '[Montant]', result, flags=re.IGNORECASE)
    result = re.sub(r'\b\d+(?:[.,]\d{2,3})*\s*(?:euros?|€)\b', '[Montant]', result, flags=re.IGNORECASE)
    
    return result, anonymizer.get_mapping_report()