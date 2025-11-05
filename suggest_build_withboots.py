import torch
import pandas as pd
import torch.nn as nn

class BootsModel(nn.Module):
    def __init__(self, num_champs, embed_dim, hidden_dim, output):
        super(BootsModel, self).__init__()
        self.embed = nn.Embedding(num_champs, embed_dim)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output)

    def forward(self, enemy_champs):
        champion_embeddings = self.embed(enemy_champs)
        x = champion_embeddings.mean(dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class BuildReccomender(nn.Module):
    def __init__(self, num_champs, embed_dim, hidden_dim, output):
        super(BuildReccomender, self).__init__()
        self.embed = nn.Embedding(num_champs, embed_dim)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output)

    def forward(self, enemy_champs):
        champion_embeddings = self.embed(enemy_champs)
        x = champion_embeddings.mean(dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

def suggest_items(adc, sup):
    enemy_adc = adc.lower()
    enemy_sup = sup.lower()
    adc = champ_data[champ_data['Champion'] == enemy_adc]
    sup = champ_data[champ_data['Champion'] == enemy_sup]
    damage = adc['Damage'].values[0] + sup['Damage'].values[0]
    toughness = adc['Toughness'].values[0] + sup['Toughness'].values[0]
    cc = adc['CC'].values[0] + sup['CC'].values[0]
    mobility = adc['Mobility'].values[0] + sup['Mobility'].values[0]
    input_values = [damage, toughness, cc, mobility]
    boots_model.eval()
    model.eval()
    with torch.no_grad():
        enemy_input = torch.tensor([input_values], dtype=torch.long)
        preds_boots = boots_model(enemy_input)

        probs = preds_boots.tolist()[0]
        item_prob = {}
        for i in range(num_boots):
            item_prob[SIMPLIFIED_BOOTS_LIST[i]] = probs[i]
        sorted_item_probs = sorted(item_prob.items(), key=lambda item: item[1], reverse=True)
        predicted_boots = sorted_item_probs[0][0]

        # convert to one-hot
        max_index = torch.argmax(preds_boots).item()
        one_hot = [1 if i == max_index else 0 for i in range(len(SIMPLIFIED_BOOTS_LIST))]

        for boots_encoded in one_hot:
            input_values.append(boots_encoded)

        enemy_input = torch.tensor([input_values], dtype=torch.long)
        preds = model(enemy_input)
    probs = preds.tolist()[0]
    item_prob = {}
    for i in range(num_items):
        item_prob[LEGENDARY_ITEM_LIST[i]] = probs[i]
    sorted_item_probs = sorted(item_prob.items(), key=lambda item: item[1], reverse=True)
    return predicted_boots, sorted_item_probs[:5]

champ_data = pd.read_excel("LeagueChampsData.xlsx")
champ_data['Champion'] = champ_data['Champion'].apply(lambda x: x.lower())
SUPPORT_ITEM_LIST = ["Bloodsong","Celestial Opposition","Dream Maker","Solstice Sleigh","Zaz'Zak's Realmspike"]
LEGENDARY_ITEM_LIST = ["Abyssal Mask","Archangel's Staff","Ardent Censer","Axiom Arc","Banshee's Veil","Black Cleaver",
                      "Blackfire Torch","Blade of the Ruined King","Bloodletter's Curse","Bloodthirster","Chempunk Chainsword",
                      "Cosmic Drive","Cryptbloom","Dawncore","Dead Man's Plate","Death's Dance","Echoes of Helia","Eclipse","Edge of Night",
                      "Essence Reaver","Experimental Hexplate","Fimbulwinter","Force of Nature","Frozen Heart","Guardian Angel","Guinsoo's Rageblade",
                      "Heartsteel","Hextech Rocketbelt","Hollow Radiance","Horizon Focus","Hubris","Hullbreaker","Iceborn Gauntlet",
                       "Immortal Shieldbow","Imperial Mandate","Infinity Edge","Jak'Sho, The Protean","Kaenic Rookern","Knight's Vow",
                      "Kraken Slayer","Liandry's Torment","Lich Bane","Locket of the Iron Solari","Lord Dominik's Regards","Luden's Companion",
                      "Malignance","Manamune","Maw of Malmortius","Mejai's Soulstealer","Mercurial Scimitar","Mikael's Blessing",
                      "Moonstone Renewer","Morellonomicon","Mortal Reminder","Muramana","Nashor's Tooth","Navori Flickerblade",
                      "Opportunity","Overlord's Bloodmail","Phantom Dancer","Profane Hydra","Rabadon's Deathcap","Randiun's Omen",
                      "Rapid Firecannon","Ravenous Hydra","Redemption","Riftmaker","Rod of Ages","Runaan's Hurricane","Rylai's Crystal Scepter",
                      "Seraph's Embrace","Serpent's Fang","Serylda's Grudge","Shadowflame","Shurelya's Battlesong","Spear of Shojin",
                      "Spirit's Visage","Staff of Flowing Water","Statikk Shiv","Sterak's Gage","Stormsurge","Stridebreaker",
                      "Sundered Sky","Sunfire Aegis","Terminus","The Collector","Thornmail","Titanic Hydra","Trailblazer","Trinity Force",
                      "Umbral Glaive","Unending Despair","Vigilant Wardstone","Void Staff","Voltaic Cyclosword","Warmog's Armor",
                      "Winter's Approach","Wit's End","Youmuu's Ghostblade","Yun Tal Wildarrows","Zeke's Convergence","Zhonya's Hourglass"]
BOOTS_LIST = ["Berserker's Greaves","Gunmetal Greaves","Mercury's Treads","Chainlaced Crushers",
              "Ionian Boots of Lucidity","Crimson Lucidity","Boots of Swiftness","Swiftmarch","Plated Steelcaps","Armored Advance",
              "Sorcerer's Shoes","Spellslinger's Shoes","Symbiotic Soles","Synchronized Souls","Forever Forward"]
SIMPLIFIED_BOOTS_LIST = ["Berserker's Greaves","Mercury's Treads","Ionian Boots of Lucidity",
                         "Boots of Swiftness","Plated Steelcaps","Sorcerer's Shoes","Symbiotic Soles"]

SUPPORT_ITEM_LIST = [item.lower() for item in SUPPORT_ITEM_LIST]
LEGENDARY_ITEM_LIST = [item.lower() for item in LEGENDARY_ITEM_LIST]
BOOTS_LIST = [item.lower() for item in BOOTS_LIST]
SIMPLIFIED_BOOTS_LIST = [item.lower() for item in SIMPLIFIED_BOOTS_LIST]

all_champs = champ_data["Champion"].tolist()
champ_dict = {champ.lower():i for i,champ in enumerate(all_champs)}
item_dict = {item:i for i, item in enumerate(LEGENDARY_ITEM_LIST)}

num_champs = len(all_champs)
num_items = len(LEGENDARY_ITEM_LIST)
num_boots = len(SIMPLIFIED_BOOTS_LIST)

boots_embed_dim = 32
boots_hidden_dim = 32
embed_dim = 64
hidden_dim = 64

boots_model = BootsModel(num_champs, boots_embed_dim, boots_hidden_dim, num_boots)
boots_model.load_state_dict(torch.load("boots_model.pth", map_location=torch.device('cpu')))

model = BuildReccomender(num_champs, embed_dim, hidden_dim, num_items)
model.load_state_dict(torch.load("model_withboots.pth", map_location=torch.device('cpu')))