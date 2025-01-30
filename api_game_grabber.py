import requests
from datetime import datetime
import time
import numpy as np
import os

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

SAMPLES_PER_MATCH = 20  # Nombre de datapoints générés par match
MASK_PROB = 0.6         # Probabilité de masquer un champion
MASK_VALUE = len(CHAMPIONS)  # Valeur spéciale pour les slots vides



API_KEY = "RGAPI-5bdf7895-146f-47a9-ae81-a1ad7463ce8f"  # Remplacez par votre clé API
BASE_URLS = {
    "EUROPE": "https://europe.api.riotgames.com",
    "EUW1": "https://euw1.api.riotgames.com"
}

original_id_to_idx = {str(champ[0]): idx for idx, champ in enumerate(CHAMPIONS)}
n_champions = len(CHAMPIONS)
MASK_VALUE = n_champions
DATA_FILE = "lol_draft_data_v2.npz"  # Nouveau fichier pour la nouvelle structure

BASE_URLS = {
    "EUROPE": "https://europe.api.riotgames.com",
    "EUW1": "https://euw1.api.riotgames.com"
}

def get_match_details(match_id):
    """Récupère les détails du match avec la nouvelle structure"""
    url = f"{BASE_URLS['EUROPE']}/lol/match/v5/matches/{match_id}"
    response = requests.get(url, headers={"X-Riot-Token": API_KEY})
    data = response.json()
    
    winner = None
    teams = {}

    if data["info"]["gameMode"] != "CLASSIC":
        print("aya")
        return 0
    
    for team in data["info"]["teams"]:
        if team["teamId"] == 100:
            teams["blue"] = [p["championId"] for p in data["info"]["participants"] if p["teamId"] == 100]
        if team["teamId"] == 200:
            teams["red"] = [p["championId"] for p in data["info"]["participants"] if p["teamId"] == 200]
        if team["win"]:
            winner = team["teamId"]
    
    
    return {
        "winner": winner,
        "blue_team": teams.get("blue", []),
        "red_team": teams.get("red", [])
    }

def process_match(match_data, mask_value):
    """Génère les échantillons avec l'équipe gagnante en première position"""
    samples = []
    
    try:
        # Déterminer l'équipe gagnante
        is_blue_winner = (match_data["winner"] == 100)
        win_team = match_data["blue_team"] if is_blue_winner else match_data["red_team"]
        lose_team = match_data["red_team"] if is_blue_winner else match_data["blue_team"]

        # Convertir les IDs
        win_converted = [original_id_to_idx.get(str(c), mask_value) for c in win_team]
        lose_converted = [original_id_to_idx.get(str(c), mask_value) for c in lose_team]

        for _ in range(SAMPLES_PER_MATCH):
            # Génération des masques
            win_mask = np.random.random(5) < MASK_PROB
            lose_mask = np.random.random(5) < MASK_PROB

            # Appliquer les masques
            masked_win = [c if not m else MASK_VALUE for c, m in zip(win_converted, win_mask)]
            masked_lose = [c if not m else MASK_VALUE for c, m in zip(lose_converted, lose_mask)]

            # Créer la cible
            target = np.zeros(n_champions)
            for idx, (champ, masked) in enumerate(zip(win_converted, win_mask)):
                if masked and champ != MASK_VALUE:
                    target[champ] = 1

            samples.append((masked_win, masked_lose, target))
            
    except Exception as e:
        print(f"Erreur de traitement : {str(e)}")
    
    return samples

def load_existing_data():
    """Charge les données existantes avec la nouvelle structure"""
    if os.path.exists(DATA_FILE):
        data = np.load(DATA_FILE, allow_pickle=True)
        return (
            data['X_win'].tolist(),
            data['X_lose'].tolist(),
            data['y'].tolist(),
            set(data['match_ids']),
            data['mask_value'].item()
        )
    return [], [], [], set(), MASK_VALUE

def save_data(X_win, X_lose, y, match_ids, mask_value):
    """Sauvegarde avec la nouvelle structure"""
    np.savez_compressed(
        DATA_FILE,
        X_win=np.array(X_win, dtype=np.int32),
        X_lose=np.array(X_lose, dtype=np.int32),
        y=np.array(y, dtype=np.float32),
        match_ids=np.array(list(match_ids), dtype=str),
        mask_value=np.array([mask_value])
    )

# Workflow principal
if __name__ == "__main__":
    X_win, X_lose, y, existing_matches, mask_value = load_existing_data()
    
    # Récupérer les matches de la ligue Challenger
    challenger_league = requests.get(
        f"{BASE_URLS['EUW1']}/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5",
        headers={"X-Riot-Token": API_KEY}
    ).json()

    for entry in challenger_league["entries"]:
        try:
            summoner = requests.get(
                f"{BASE_URLS['EUW1']}/lol/summoner/v4/summoners/{entry['summonerId']}",
                headers={"X-Riot-Token": API_KEY}
            ).json()
            
            match_history = requests.get(
                f"{BASE_URLS['EUROPE']}/lol/match/v5/matches/by-puuid/{summoner['puuid']}/ids",
                headers={"X-Riot-Token": API_KEY}
            ).json()

            for match_id in match_history:
                if match_id not in existing_matches:
                    match_data = get_match_details(match_id)

                    if match_data == 0:
                        print("pas classic")
                        continue

                    samples = process_match(match_data, mask_value)

                    # print(match_data)
                    # print(samples)
                    # Ajouter les nouveaux échantillons
                    for win, lose, target in samples:
                        X_win.append(win)
                        X_lose.append(lose)
                        y.append(target)
                        
                    existing_matches.add(match_id)
                    save_data(X_win, X_lose, y, existing_matches, mask_value)
                    print(f"Match {match_id} traité. Échantillons totaux: {len(X_win)}")
                    
                    time.sleep(1.2)  # Respect du rate limit
                else:
                    print("deja dedans")
                    time.sleep(1.2)
        except Exception as e:
            print(f"Erreur avec le summoner {entry['summonerId']}: {str(e)}")
            time.sleep(1.2)

    print("Collecte terminée !")
