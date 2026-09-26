from flask import Flask, request, render_template, jsonify, send_from_directory
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import io
import os

app = Flask(__name__)

# Configure paths
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), "static")
IMAGES_FOLDER = os.path.join(STATIC_FOLDER, "images")
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Load the model
model = mobilenet_v3_large(pretrained=False)
model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=25)  # Adjust for 25 bird classes
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "best_model_epoch_19.pth"), map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Bird class names and info

bird_info = {
    'Asian-Green-Bee-Eater': {
    'name': 'Asian Green Bee-Eater',
    'scientific_name': 'Merops orientalis',
    'description': 'This small bee-eater is found across tropical southern Asia, known for its vibrant green color and slender body. It primarily feeds on bees and other flying insects.',
    'image': 'asian_green_bee_eater.jpg'
},
    'Brown-Headed-Barbet': {
    'name': 'Brown-Headed Barbet',
    'scientific_name': 'Psilopogon zeylanicus',
    'description': 'A species of barbet found in India and Sri Lanka. It is known for its distinctive brown head and striking green plumage. It nests in tree cavities.',
    'image': 'brown_headed_barbet.jpg'
},
    'Cattle-Egret': {
    'name': 'Cattle Egret',
    'scientific_name': 'Bubulcus ibis',
    'description': 'A small white heron often seen in agricultural fields, typically foraging alongside cattle, buffalo, and other large mammals. Known for its symbiotic relationship with grazing animals.',
    'image': 'cattle_egret.jpg'
},
    'Common-Kingfisher': {
    'name': 'Common Kingfisher',
    'scientific_name': 'Alcedo atthis',
    'description': 'A small, brightly colored kingfisher known for its sharp beak and vibrant blue and orange plumage. It dives into water to catch fish.',
    'image': 'common_kingfisher.jpg'
},
    'Common-Myna': {
    'name': 'Common Myna',
    'scientific_name': 'Acridotheres tristis',
    'description': 'A highly adaptable bird found across Asia. It thrives in urban and suburban areas, known for its loud calls and mimicry abilities.',
    'image': 'common_myna.jpg'
},
    'Common-Rosefinch': {
    'name': 'Common Rosefinch',
    'scientific_name': 'Carpodacus erythrinus',
    'description': 'A small, seed-eating bird found across Europe and Asia. The males are known for their rosy-red plumage, especially during the breeding season.',
    'image': 'common_rosefinch.jpg'
},
    'Common-Tailorbird': {
    'name': 'Common Tailorbird',
    'scientific_name': 'Orthotomus sutorius',
    'description': 'A small warbler-like bird, famous for its unique ability to stitch leaves together to create its nest. Found in gardens and shrublands across Asia.',
    'image': 'common_tailorbird.jpg'
},
    'Coppersmith-Barbet': {
    'name': 'Coppersmith Barbet',
    'scientific_name': 'Psilopogon haemacephalus',
    'description': 'A small bird with a metallic "tuk-tuk-tuk" call, reminiscent of a coppersmith at work. It has a distinctive red forehead and throat.',
    'image': 'coppersmith_barbet.jpg'
},
    'Forest-Wagtail': {
    'name': 'Forest Wagtail',
    'scientific_name': 'Dendronanthus indicus',
    'description': 'A migratory bird that inhabits forests, known for its distinct tail-wagging behavior. It can often be seen foraging on the forest floor for insects.',
    'image': 'forest_wagtail.jpg'
},
    'Gray-Wagtail': {
    'name': 'Gray Wagtail',
    'scientific_name': 'Motacilla cinerea',
    'description': 'A slender, long-tailed bird commonly found near rivers and streams. It has a distinctive yellow underbelly and constantly wags its tail.',
    'image': 'gray_wagtail.jpg'
},
    'Hoopoe': {
    'name': 'Hoopoe',
    'scientific_name': 'Upupa epops',
    'description': 'A striking bird known for its long, curved beak and its vibrant crest of feathers. It feeds on insects and grubs, often found in open areas.',
    'image': 'hoopoe.jpg'
},
    'House-Crow': {
    'name': 'House Crow',
    'scientific_name': 'Corvus splendens',
    'description': 'A highly social bird commonly found in urban areas. It is an opportunistic feeder and is known for its intelligence and adaptability.',
    'image': 'house_crow.jpg'
},
    'Indian-Grey-Hornbill': {
    'name': 'Indian Grey Hornbill',
    'scientific_name': 'Ocyceros birostris',
    'description': 'A large bird with a prominent curved bill, often found in forests and urban areas. It feeds on fruits, small reptiles, and insects.',
    'image': 'indian_grey_hornbill.jpg'
},
    'Indian-Peacock': {
    'name': 'Indian Peacock',
    'scientific_name': 'Pavo cristatus',
    'description': 'A stunningly beautiful bird known for its iridescent blue-green plumage and spectacular tail feathers in males, which are used in courtship displays.',
    'image': 'indian_peacock.jpg'
},
    'Indian-Pitta': {
    'name': 'Indian Pitta',
    'scientific_name': 'Pitta brachyura',
    'description': 'A brightly colored bird with a stout body and a distinctive call. It is often found on the forest floor searching for insects.',
    'image': 'indian_pitta.jpg'
},
    'Indian-Roller': {
    'name': 'Indian Roller',
    'scientific_name': 'Coracias benghalensis',
    'description': 'A medium-sized bird with brilliant blue wings and a rolling flight display. It is commonly seen perched on wires or poles in open countryside.',
    'image': 'indian_roller.jpg'
},
    'Jungle-Babbler': {
    'name': 'Jungle Babbler',
    'scientific_name': 'Turdoides striata',
    'description': 'A social bird often found in noisy flocks, known as the "Seven Sisters" due to its gregarious nature. It is found in woodlands and gardens.',
    'image': 'jungle_babbler.jpg'
},
    'Northern-Lapwing': {
    'name': 'Northern Lapwing',
    'scientific_name': 'Vanellus vanellus',
    'description': 'A wader with a distinctive crest and a dramatic black-and-white wing pattern. It is often found in wetlands and marshy areas.',
    'image': 'northern_lapwing.jpg'
},
    'Red-Wattled-Lapwing': {
    'name': 'Red-Wattled Lapwing',
    'scientific_name': 'Vanellus indicus',
    'description': 'A noisy bird, often seen near water bodies. Its name comes from the red wattles on either side of its beak, which are prominent features.',
    'image': 'red_wattled_lapwing.jpg'
},
    'Ruddy-Shelduck': {
    'name': 'Ruddy Shelduck',
    'scientific_name': 'Tadorna ferruginea',
    'description': 'A large, orange-brown duck often found in pairs. It prefers large, open wetlands and is known for its loud calls.',
    'image': 'ruddy_shelduck.jpg'
},
    'Rufous-Treepie': {
    'name': 'Rufous Treepie',
    'scientific_name': 'Dendrocitta vagabunda',
    'description': 'A member of the crow family, this bird is easily recognizable due to its bright orange and black plumage. It is found in forests and gardens.',
    'image': 'rufous_treepie.jpg'
},
    'Sarus-Crane': {
    'name': 'Sarus Crane',
    'scientific_name': 'Antigone antigone',
    'description': 'The tallest flying bird in the world, known for its striking red head and gray body. It inhabits wetlands and is often seen in pairs.',
    'image': 'sarus_crane.jpg'
},
    'White-Breasted-Kingfisher': {
    'name': 'White-Breasted Kingfisher',
    'scientific_name': 'Halcyon smyrnensis',
    'description': 'A large kingfisher with a vibrant blue body, brown head, and a distinctive white throat. It is found near water but also inhabits woodlands and urban areas.',
    'image': 'white_breasted_kingfisher.jpg'
},
    'White-Breasted-Waterhen': {
    'name': 'White-Breasted Waterhen',
    'scientific_name': 'Amaurornis phoenicurus',
    'description': 'A marsh bird with a white face and underparts, and a dark brown back. It is often seen foraging in wetlands and rice fields.',
    'image': 'white_breasted_waterhen.jpg'
},
    'White-Wagtail': {
    'name': 'White Wagtail',
    'scientific_name': 'Motacilla alba',
    'description': 'A slender bird known for its black, white, and grey plumage and its constant tail-wagging behavior. Often seen near water, searching for insects.',
    'image': 'white_wagtail.jpg'
},

    # Add more bird details here...
}


    # Add other bird details here...


# List of bird class names
class_names = list(bird_info.keys())

@app.route('/')
def index():
    return render_template('index.html', birds=bird_info.values())

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open the image and preprocess
        img = Image.open(io.BytesIO(file.read())).convert('RGB')  # grayscale/RGBA -> 3 channels
        img = transform(img).unsqueeze(0).to(device)

        # Make the prediction
        with torch.no_grad():
            probs = torch.softmax(model(img), dim=1)[0]
        top_probs, top_idx = probs.topk(3)

        bird_details = bird_info[class_names[top_idx[0].item()]]

        # Return prediction details as JSON
        return jsonify({
            "prediction": bird_details['name'],
            "scientific_name": bird_details['scientific_name'],
            "description": bird_details['description'],
            "image": bird_details['image'],
            "confidence": round(top_probs[0].item(), 4),
            "top3": [{"name": bird_info[class_names[i]]['name'], "confidence": round(p, 4)}
                     for p, i in zip(top_probs.tolist(), top_idx.tolist())]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/images/<filename>')
def static_images(filename):
    # Serve images from the static/images folder
    return send_from_directory(IMAGES_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
