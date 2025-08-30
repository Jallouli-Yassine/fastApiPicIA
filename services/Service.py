import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import torch
from dotenv import load_dotenv
import io
import time
import random

load_dotenv()  # Charge les variables d'environnement depuis .env


class Service:
    def __init__(self):
        # Vérifier si CUDA est disponible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # BLIP model pour les descriptions d'images
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=False)
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)

        # T5 model pour le traitement de texte
        self.modelText = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

    def process_text(self, input_text):
        """
        Traite le texte d'entrée en utilisant le modèle T5
        """
        try:
            # Tokenize input text
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

            # Generate output using the model
            outputs = self.modelText.generate(input_ids)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return result
        except Exception as e:
            raise Exception(f"Error processing text: {str(e)}")

    def generate_caption(self, image_file):
        """
        Génère une légende pour l'image en utilisant le modèle BLIP
        """
        try:
            # Start timer for processing
            start_time = time.time()

            # Process the image and generate a description
            textLaravel = "garden of"

            # Open the uploaded file as an image
            image = Image.open(image_file).convert('RGB')

            # Process the image for the model
            inputs = self.processor(image, textLaravel, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)

            # Log processing time
            elapsed_time = time.time() - start_time
            print(f"Processing time: {elapsed_time:.2f} seconds")

            return caption
        except Exception as e:
            raise Exception(f"Error generating caption: {str(e)}")

    def generate_team_logo_descriptions(self, image_file, team_name, num_suggestions=3):
        try:
            start_time = time.time()

            # Obtenir la description basique de l'image avec BLIP
            image = Image.open(image_file).convert('RGB')

            # Utiliser un prompt plus approprié pour un logo
            logo_prompt = "logo representing"

            # Générer une description basique de l'image
            inputs = self.processor(image, logo_prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=30)
            basic_caption = self.processor.decode(out[0], skip_special_tokens=True)

            print("basic caption: ", basic_caption)

            # Extraire les éléments clés de la description
            key_elements = basic_caption.replace(logo_prompt, "").strip()

            # Nettoyage des [UNK] et caractères spéciaux
            import re
            key_elements = re.sub(r'\[UNK\]', '', key_elements)
            key_elements = key_elements.strip()

            # Si après nettoyage la description est vide ou trop courte, utiliser une description générique
            if len(key_elements) < 5:
                key_elements = "un emblème puissant"

            print("key elements after cleaning: ", key_elements)

            # Phrases d'inspiration pour les équipes
            inspirational_templates = [
                "Une équipe qui incarne {elements}, prête à dominer l'arène avec stratégie et détermination.",
                "Représentant {elements}, cette équipe est forgée dans l'esprit de compétition et d'excellence.",
                "Avec {elements} comme emblème, cette formation est déterminée à marquer l'histoire des compétitions.",
                "Portant fièrement {elements}, l'équipe s'élance vers la victoire avec passion et courage.",
                "L'esprit de {elements} guide cette équipe vers les sommets de la gloire et du succès.",
                "Symbole de puissance avec {elements}, cette formation redoutable ne connaît pas la défaite.",
                "Incarnant {elements}, les membres de cette équipe brillent par leur talent et leur cohésion.",
                "Cette formation élite représente {elements}, synonyme d'excellence dans chaque compétition.",
                "Portant l'emblème de {elements}, cette équipe inspire respect et admiration à ses adversaires."
            ]

            # Ajouter des templates avec le nom d'équipe
            team_templates = [
                f"L'équipe {team_name}, arborant {key_elements}, est prête à conquérir tous les défis.",
                f"Avec {key_elements} comme symbole, {team_name} s'impose comme une force incontournable.",
                f"{team_name}, représentant {key_elements}, est une équipe dont la légende ne fait que commencer.",
                f"Les champions de {team_name} portent fièrement {key_elements}, symbole de leur unité et détermination."
            ]
            # Combiner les deux ensembles de templates
            all_templates = inspirational_templates + team_templates

            # Mélanger les templates pour une sélection aléatoire
            random.shuffle(all_templates)

            # Sélectionner le nombre demandé de templates (ou tous si pas assez)
            selected_templates = all_templates[:min(num_suggestions, len(all_templates))]

            # Si nous n'avons pas assez de templates, réutiliser certains avec des variations
            while len(selected_templates) < num_suggestions:
                template = random.choice(all_templates)
                selected_templates.append(template)

            # Créer les descriptions finales
            descriptions = []
            for template in selected_templates:
                if "{elements}" in template:
                    description = template.format(elements=key_elements)
                else:
                    description = template  # Pour les f-strings qui contiennent déjà key_elements
                descriptions.append(description)

            # Log processing time
            elapsed_time = time.time() - start_time
            print(f"Team logo descriptions processing time: {elapsed_time:.2f} seconds")

            return descriptions

        except Exception as e:
            print(f"Error in generate_team_logo_descriptions: {str(e)}")
            raise Exception(f"Error generating team logo descriptions: {str(e)}")


    def generate_team_logo_description(self, image_file, team_name):
        descriptions = self.generate_team_logo_descriptions(image_file, team_name, num_suggestions=1)
        return descriptions[0]