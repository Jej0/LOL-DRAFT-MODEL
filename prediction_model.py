import numpy as np
import tensorflow as tf
from difflib import get_close_matches
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import keras_cv
# from tf_embedding import AttentionBlock
from tf_embedding import AttentionBlock
# Liste complète des champions avec leurs IDs et noms
CHAMPIONS = [
    (266, "Aatrox"),
    (103, "Ahri"),
    (84, "Akali"),
    (166, "Akshan"),
    (12, "Alistar"),
    (799, "Ambessa"),
    (32, "Amumu"),
    (34, "Anivia"),
    (1, "Annie"),
    (523, "Aphelios"),
    (22, "Ashe"),
    (136, "Aurelion Sol"),
    (893, "Aurora"),
    (268, "Azir"),
    (432, "Bard"),
    (200, "Bel'Veth"),
    (53, "Blitzcrank"),
    (63, "Brand"),
    (201, "Braum"),
    (233, "Briar"),
    (51, "Caitlyn"),
    (164, "Camille"),
    (69, "Cassiopeia"),
    (31, "Cho'Gath"),
    (42, "Corki"),
    (122, "Darius"),
    (131, "Diana"),
    (119, "Draven"),
    (36, "Dr. Mundo"),
    (245, "Ekko"),
    (60, "Elise"),
    (28, "Evelynn"),
    (81, "Ezreal"),
    (9, "Fiddlesticks"),
    (114, "Fiora"),
    (105, "Fizz"),
    (3, "Galio"),
    (41, "Gangplank"),
    (86, "Garen"),
    (150, "Gnar"),
    (79, "Gragas"),
    (104, "Graves"),
    (887, "Gwen"),
    (120, "Hecarim"),
    (74, "Heimerdinger"),
    (910, "Hwei"),
    (420, "Illaoi"),
    (39, "Irelia"),
    (427, "Ivern"),
    (40, "Janna"),
    (59, "Jarvan IV"),
    (24, "Jax"),
    (126, "Jayce"),
    (202, "Jhin"),
    (222, "Jinx"),
    (145, "Kai'Sa"),
    (429, "Kalista"),
    (43, "Karma"),
    (30, "Karthus"),
    (38, "Kassadin"),
    (55, "Katarina"),
    (10, "Kayle"),
    (141, "Kayn"),
    (85, "Kennen"),
    (121, "Kha'Zix"),
    (203, "Kindred"),
    (240, "Kled"),
    (96, "Kog'Maw"),
    (897, "K'Sante"),
    (7, "LeBlanc"),
    (64, "Lee Sin"),
    (89, "Leona"),
    (876, "Lillia"),
    (127, "Lissandra"),
    (236, "Lucian"),
    (117, "Lulu"),
    (99, "Lux"),
    (54, "Malphite"),
    (90, "Malzahar"),
    (57, "Maokai"),
    (11, "Master Yi"),
    (800, "Mel"),
    (902, "Milio"),
    (21, "Miss Fortune"),
    (62, "Wukong"),
    (82, "Mordekaiser"),
    (25, "Morgana"),
    (950, "Naafiri"),
    (267, "Nami"),
    (75, "Nasus"),
    (111, "Nautilus"),
    (518, "Neeko"),
    (76, "Nidalee"),
    (895, "Nilah"),
    (56, "Nocturne"),
    (20, "Nunu & Willump"),
    (2, "Olaf"),
    (61, "Orianna"),
    (516, "Ornn"),
    (80, "Pantheon"),
    (78, "Poppy"),
    (555, "Pyke"),
    (246, "Qiyana"),
    (133, "Quinn"),
    (497, "Rakan"),
    (33, "Rammus"),
    (421, "Rek'Sai"),
    (526, "Rell"),
    (888, "Renata Glasc"),
    (58, "Renekton"),
    (107, "Rengar"),
    (92, "Riven"),
    (68, "Rumble"),
    (13, "Ryze"),
    (360, "Samira"),
    (113, "Sejuani"),
    (235, "Senna"),
    (147, "Seraphine"),
    (875, "Sett"),
    (35, "Shaco"),
    (98, "Shen"),
    (102, "Shyvana"),
    (27, "Singed"),
    (14, "Sion"),
    (15, "Sivir"),
    (72, "Skarner"),
    (901, "Smolder"),
    (37, "Sona"),
    (16, "Soraka"),
    (50, "Swain"),
    (517, "Sylas"),
    (134, "Syndra"),
    (223, "Tahm Kench"),
    (163, "Taliyah"),
    (91, "Talon"),
    (44, "Taric"),
    (17, "Teemo"),
    (412, "Thresh"),
    (18, "Tristana"),
    (48, "Trundle"),
    (23, "Tryndamere"),
    (4, "Twisted Fate"),
    (29, "Twitch"),
    (77, "Udyr"),
    (6, "Urgot"),
    (110, "Varus"),
    (67, "Vayne"),
    (45, "Veigar"),
    (161, "Vel'Koz"),
    (711, "Vex"),
    (254, "Vi"),
    (234, "Viego"),
    (112, "Viktor"),
    (8, "Vladimir"),
    (106, "Volibear"),
    (19, "Warwick"),
    (498, "Xayah"),
    (101, "Xerath"),
    (5, "Xin Zhao"),
    (157, "Yasuo"),
    (777, "Yone"),
    (83, "Yorick"),
    (350, "Yuumi"),
    (154, "Zac"),
    (238, "Zed"),
    (221, "Zeri"),
    (115, "Ziggs"),
    (26, "Zilean"),
    (142, "Zoe"),
    (143, "Zyra")
]
# Configuration

PHASES_RED_FIRST = [
    ('R1', 1), ('B1', 1), ('B2', 1),
    ('R2', 1), ('R3', 1), ('B3', 1), ('B4', 1),
    ('R4', 1), ('R5', 1), ('B5', 1)
]

PHASES_BLUE_FIRST = [
    ('B1', 1), ('R1', 1), ('R2', 1),
    ('B2', 1), ('B3', 1), ('R3', 1), ('R4', 1),
    ('B4', 1), ('B5', 1), ('R5', 1)
]

MASK_VALUE = len(CHAMPIONS)

# Création des mappings
original_id_to_idx = {str(c[0]): i for i, c in enumerate(CHAMPIONS)}
idx_to_original_id = {i: c[0] for i, c in enumerate(CHAMPIONS)}
name_to_id = {c[1].lower().replace("'", "").replace(" ", ""): c[0] for c in CHAMPIONS}

class DraftAssistant:
    def __init__(self, is_red_first=True):
        self.is_red_first = is_red_first
        self.phases = PHASES_RED_FIRST if is_red_first else PHASES_BLUE_FIRST
        self.current_phase = 0
        self.red_picks = []
        self.blue_picks = []
        self.model = self.load_model()
    
    def load_model(self):
        try:
            return tf.keras.models.load_model(
                'final_model_HUGE2.keras',
                custom_objects={
                    'AttentionBlock': AttentionBlock,
                    'FocalLoss': keras_cv.losses.FocalLoss
                }
            )
        except Exception as e:
            print(f"Erreur de chargement: {str(e)}")
            exit(1)

            
    def get_recommendations(self):
        """Génère les recommandations basées sur l'état actuel"""
        red = [original_id_to_idx.get(str(c), MASK_VALUE) for c in self.red_picks] + [MASK_VALUE]*(5 - len(self.red_picks))
        blue = [original_id_to_idx.get(str(c), MASK_VALUE) for c in self.blue_picks] + [MASK_VALUE]*(5 - len(self.blue_picks))
        
        probs = self.model.predict([np.array([red]), np.array([blue])], verbose=0)[0]
        
        picked = set(self.red_picks + self.blue_picks)
        return sorted(
            [(i, p) for i, p in enumerate(probs) if idx_to_original_id[i] not in picked],
            key=lambda x: x[1], 
            reverse=True
        )[:5]

    def display_recommendations(self, recommendations):
        print("\n" + "═"*50)
        print(" RECOMMANDATIONS ".center(50))
        print("═"*50)
        for idx, (champ_idx, prob) in enumerate(recommendations, 1):
            champ_id = idx_to_original_id.get(champ_idx, "Inconnu")
            champ_name = next((c[1] for c in CHAMPIONS if c[0] == champ_id), "Champion Inconnu")
            print(f"{idx}. {champ_name} ({prob*100:.1f}%)")

    def process_input(self, input_str):
        """Convertit les noms en IDs avec gestion d'erreur améliorée"""
        names = [n.strip() for n in input_str.split(',') if n.strip()]
        converted = []
        
        for name in names:
            clean_name = name.lower().replace("'", "").replace(" ", "")
            if clean_name in name_to_id:
                converted.append(name_to_id[clean_name])
            else:
                matches = get_close_matches(clean_name, name_to_id.keys(), n=3, cutoff=0.5)
                if matches:
                    suggestions = [next((c[1] for c in CHAMPIONS if c[0] == name_to_id[m]), m) for m in matches]
                    print(f"\nErreur: '{name}' non reconnu. Suggestions: {', '.join(suggestions)}")
                else:
                    print(f"\nErreur: '{name}' non trouvé dans la base de données")
                return None
        return converted

    def run_draft(self):
        print("\n" + "═"*50)
        print(f" DRAFT {'ROUGE FIRST PICK' if self.is_red_first else 'BLEU FIRST PICK'} ".center(50))
        print("═"*50)

        for phase, count in self.phases:
            team = 'red' if phase.startswith('R') else 'blue'
            
            # Affichage des recommandations avant les choix rouge
            if team == 'red':
                recos = self.get_recommendations()
                self.display_recommendations(recos)
                self.display_state()

            # Gestion de l'entrée utilisateur
            while True:
                input_str = input(f"\nPhase {phase} ▶ Entrez {count} champion(s): ")
                champion_ids = self.process_input(input_str)
                
                if champion_ids and len(champion_ids) == count:
                    break
                print(f"Veuillez entrer exactement {count} champion(s) valide(s)")

            # Mise à jour des picks
            if team == 'red':
                self.red_picks.extend(champion_ids)
            else:
                self.blue_picks.extend(champion_ids)

    def display_state(self):
        print("\n" + "═"*50)
        print(" ÉTAT ACTUEL ".center(50))
        print(f"Rouge: {[self.id_to_name(c) for c in self.red_picks]}")
        print(f"Bleu:  {[self.id_to_name(c) for c in self.blue_picks]}")
        print("═"*50)

    def id_to_name(self, champ_id):
        return next((c[1] for c in CHAMPIONS if c[0] == champ_id), "Inconnu")

if __name__ == "__main__":
    is_red_first = input("L'équipe rouge est-elle first pick ? (o/n) ").lower().startswith('o')
    assistant = DraftAssistant(is_red_first)
    assistant.run_draft()