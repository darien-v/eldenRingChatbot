import os
import torch
from model import GPTConfig, GPT
import tiktoken

# configuration (based on sample.py and sample_funcom.py)
init_from = 'resume'  # assuming you're resuming from a trained model
out_dir = 'out'
temperature = 0.8
top_k = 200
device = 'cuda'
dtype = 'bfloat16'
num_samples = 3 # set to 1 since we generate one review per description
max_new_tokens = 500  



# set up the device, model, and encoding
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.no_grad()  # Use no_grad context for inference

if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)

# assuming GPT-2 encodings
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

#context = """Phantom Great Rune\nType\nGreat Rune\nEffect\nGive blessing of blood to enemies in invaded world\nPhantom Great Rune is a Great Rune, Multiplayer Item, and Consumable in Elden Ring. Phantom Great Rune gives blessing of blood to enemies in invaded world. Great Runes in Elden Ring are special items dropped by Demigod Bosses that can be equipped to acquire special passive bonuses.\nItem for online play. Obtained after using Mohg's Great Rune and invading another world. Consumed upon use.\nEnemies inside the invaded world receive a blessing of blood, which boosts their attack power when blood loss occurs nearby. Also, your HP recovers when blessed enemies defeat a player.\nWhere to find Phantom Great Rune in Elden Ring\nUnlike other great runes, these are consumable (they recharge every invasion), and are not directly gained through completing a questline nor killing Shardbearers, and are obtained through Mohg's Great Rune as an online (PVP) only item.\n\nElden Ring Phantom Great Rune Use\nIf you want to use the Phantom Great Rune, you will need to invade a Host of Finger's world while Mohg's Great Rune is active through a Rune Arc.Upon joining the Host's world, you are granted three consumable Phantom Great Runes that will grant the Host of Finger's enemies a buff to their attack when blood loss occurs.As an invader the player will receive a -13.5% HP penalty after use while getting a +10% lifesteal boost and a +20% damage boost for 20 seconds after blood loss occured.\n\nElden Ring Phantom Great Rune Notes & Tips\nInvaders will need to rely on enemies with already existing bleed capabilities, or must use their own methods to inflict blood loss on players for the rune to have any effect.\n\nElden Ring Great Runes\nGodrick's Great Rune ♦ Great Rune of the Unborn ♦ Malenia's Great Rune ♦ Mending Rune of Perfect Order ♦ Mending Rune of the Death-Prince ♦ Mending Rune of the Fell Curse ♦ Mohg's Great Rune ♦ Morgott's Great Rune ♦ Radahn's Great Rune ♦ Rykard's Great Rune"""
#question = "Where can the Phantom Great Rune be found in Elden Ring?"

context = "\n\n Firebone Arrow (Fletched) \n\nAttack Power \n\nPhy 10\n\nMag 0 \n\nFire 90 \n\nLigt 0 \n\nHoly 0 \n\nCrit 100 \n\nPassive\n\n        - \n\nArrow\nPierce\n\nFirebone Arrow (Fletched) is an Arrow in Elden Ring. Firebone Arrow (Fletched) is a craftable ammo that can be used to inflict ranged fire damage. It can also pierce enemies' armor and the addition of fletching enables more precise shots that can travel farther. Ammunition can be used in ranged weapons such as Bows and Crossbows, so players can deal ranged damage to Enemies and Bosses.\n\nArrow whittled from animal bones. The tip is set alight before firing. Deals fire damage.\nCraftable item. The fletching adds distance to the arrow's flight.\n\nWhere to Find Firebone Arrow (Fletched) in Elden Ring\nFirebone Arrow (Fletched) can be found at the following location:\n\nCrafting\n\nElden Ring Firebone Arrow (Fletched) Notes & Tips\n\nThe feather fletching adds distance and accuracy to the standard Firebone Arrow\nYou can hold up to 99 Bone Arrow (Fletched)\nYou can store up to 600 Bone Arrow (Fletched)\nSell Value:  1\n\n Firebone Arrow (Fletched) Crafting Guide in Elden Ring\nTo craft Firebone Arrow (Fletched) (x10) you would need the Armorer's Cookbook [2] as well as the following Crafting Materials:\n\nThin Animal Bones x3\nSmoldering Butterfly x1\nFlight Pinion x1\n\nElden Ring Firebone Arrow (Fletched) Moveset & Videos\n\nVideos for the Firebone Arrow (Fletched) Coming Soon\n\nElden Ring Arrows\n\nArrow  ♦  Ballista Bolt  ♦  Black-Key Bolt  ♦  Bloodbone Arrow  ♦  Bloodbone Arrow (Fletched)  ♦  Bloodbone Bolt  ♦  Bolt  ♦  Bone Arrow  ♦  Bone Arrow (Fletched)  ♦  Bone Ballista Bolt  ♦  Bone Bolt  ♦  Bone Great Arrow  ♦  Bone Great Arrow (Fletched)  ♦  Burred Bolt  ♦  Coldbone Arrow  ♦  Coldbone Arrow (Fletched)  ♦  Coldbone Bolt  ♦  Dwelling Arrow  ♦  Explosive Bolt  ♦  Explosive Greatbolt  ♦  Fire Arrow  ♦  Firebone Arrow  ♦  Firebone Bolt  ♦  Flaming Bolt  ♦  Golden Arrow  ♦  Golden Bolt  ♦  Golden Great Arrow  ♦  Golem's Great Arrow  ♦  Golem's Magic Arrow  ♦  Great Arrow  ♦  Haligbone Arrow  ♦  Haligbone Arrow (Fletched)  ♦  Haligbone Bolt  ♦  Lightning Greabolt  ♦  Lightning Greatbolt  ♦  Lightningbone Arrow  ♦  Lightningbone Arrow (Fletched)  ♦  Lightningbone Bolt  ♦  Lordsworn's Bolt  ♦  Magicbone Arrow  ♦  Magicbone Arrow (Fletched)  ♦  Magicbone Bolt  ♦  Meteor Bolt  ♦  Perfumer's Bolt  ♦  Poisonbone Arrow  ♦  Poisonbone Arrow (Fletched)  ♦  Poisonbone Bolt  ♦  Radahn's Spear  ♦  Rainbow Stone Arrow  ♦  Rainbow Stone Arrow (Fletched)  ♦  Rotbone Arrow  ♦  Rotbone Arrow (Fletched)  ♦  Rotbone Bolt  ♦  Serpent Arrow  ♦  Shattershard Arrow  ♦  Shattershard Arrow (Fletched)  ♦  Sleepbone Arrow  ♦  Sleepbone Arrow (Fletched)  ♦  Sleepbone Bolt  ♦  Spiritflame Arrow  ♦  Storm Arrow  ♦  Stormwing Bone Arrow\n\n"
question = "How do you craft the firebone arrow?"

eos_token = "<|endoftext|>"

for i in range(num_samples):

    prompt = f"CONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER\n"
    input_ids = encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    with ctx:  # generation happens here
        generated_ids = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        generated_text = decode(generated_ids[0].tolist())
        # extracting review text after "REVIEW" and before "eos_token" (if present)
        review = generated_text.split('ANSWER')[-1].split(eos_token)[0].strip()
    
        print("-------------------------------------------------------------------------------")
        print(generated_text)
        print("=================")
        print(review)


