# Pipeline de Détection de Vols à l'Étalage (OCSORT + YOLONAS + X3D)

Ce pipeline utilise trois modèles pour détecter les actions de vol à l'étalage dans un environnement de magasin :

1. **Détection des Personnes :**
   - Localisation précise des personnes avec enregistrement de chaque position.
   
2. **Tracking et Identification :**
   - Attribution d'un identifiant unique à chaque personne à l'aide d'un tracker.

3. **Analyse des Actions :**
   - Utilisation du modèle X3D pour analyser les actions effectuées par chaque personne après 25 images.

Ces étapes permettent de surveiller et d'analyser les comportements suspects dans le contexte de la sécurité en magasin.

## Utilisation

Pour utiliser le pipeline, suivez ces étapes :

1. Naviguez vers le répertoire `ocsort` :
   cd ocsort
2. Installez les dépendances à partir du fichier requirements.txt 
   pip install -r requirements.txt
3. Veuillez ajouter le chemin de la vidéo désirée dans les deux dernières lignes du fichier

4. Exécutez le fichier pipeline.py
   python pipeline.py
