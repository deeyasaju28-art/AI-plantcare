"""
AI Plant Care Knowledge Base
Contains botanical and care data for common garden and indoor plants in English.
"""

PLANTS_DB = {
    "Holy Basil (Tulsi)": {
        "scientific_name": "Ocimum tenuiflorum",
        "category": "Medicinal Herb",
        "base_water_ml": 250,
        "ideal_sunlight": "Direct Sunlight (4-6 hours)",
        "sunlight_hours": 5,
        "growth_stages": {
            "Seedling": (1, 20),
            "Vegetative": (21, 60),
            "Flowering / Budding": (61, 120),
            "Mature": (121, 999)
        },
        "soil": "Well-draining loamy soil enriched with organic compost",
        "fertilizer": "Organic vermicompost or diluted cow dung manure every 2 weeks",
        "summer_care": "Water daily in the morning during peak summer heat. Ensure topsoil stays evenly moist.",
        "monsoon_care": "Ensure container drainage holes are clear. Avoid waterlogging around root zones to prevent fungal rot.",
        "winter_care": "Water only when the top 1 inch of soil feels dry. Shelter from dry, chilly winds.",
        "tips": "Regularly pinch off flower buds (Manjari) to stimulate lush, bushy leaf branching."
    },
    "Curry Leaf Plant": {
        "scientific_name": "Murraya koenigii",
        "category": "Aromatic Culinary Shrub",
        "base_water_ml": 400,
        "ideal_sunlight": "Full to partial direct sunlight (5-6 hours)",
        "sunlight_hours": 6,
        "growth_stages": {
            "Seedling / Sapling": (1, 30),
            "Vegetative Branching": (31, 90),
            "Foliage Expansion": (91, 180),
            "Mature Shrub": (181, 999)
        },
        "soil": "Slightly acidic, fertile red or garden soil with good aeration",
        "fertilizer": "Sour buttermilk diluted with water or nitrogen-rich compost once a month",
        "summer_care": "Water every 1 to 2 days thoroughly. Mulch around the base with dry leaves to retain moisture.",
        "monsoon_care": "Allow natural rain, but ensure water does not stagnate. Check leaves for leaf-roller caterpillars.",
        "winter_care": "Plant enters semi-dormancy in cold months. Reduce watering to twice a week.",
        "tips": "Apply diluted sour buttermilk monthly; it acts as a natural nitrogen stimulant and pest deterrent."
    },
    "Tomato": {
        "scientific_name": "Solanum lycopersicum",
        "category": "Fruiting Vegetable",
        "base_water_ml": 500,
        "ideal_sunlight": "Direct full sunlight (6-8 hours)",
        "sunlight_hours": 7,
        "growth_stages": {
            "Seedling": (1, 15),
            "Vegetative Growth": (16, 40),
            "Flowering": (41, 65),
            "Fruiting & Ripening": (66, 120)
        },
        "soil": "Rich, porous, fertile loamy soil packed with organic matter",
        "fertilizer": "Bone meal, vermicompost, and potassium/potash booster during flower and fruit set",
        "summer_care": "Maintain consistent moisture to prevent blossom-end rot and flower drop. Water at the soil level.",
        "monsoon_care": "Keep foliage dry to prevent early blight and bacterial wilt. Stake tall vines securely.",
        "winter_care": "Needs maximum sun exposure. Water only in the late morning on sunny days.",
        "tips": "Always water at the base of the stem; avoid splashing water on leaves to prevent fungal leaf spot."
    },
    "Chilli Pepper": {
        "scientific_name": "Capsicum annuum",
        "category": "Spice / Vegetable",
        "base_water_ml": 350,
        "ideal_sunlight": "Direct sunlight (5-6 hours)",
        "sunlight_hours": 6,
        "growth_stages": {
            "Seedling": (1, 20),
            "Vegetative Growth": (21, 50),
            "Flowering": (51, 75),
            "Harvesting / Fruiting": (76, 180)
        },
        "soil": "Well-drained sandy loam with perlite or organic compost",
        "fertilizer": "Neem cake powder, organic poultry manure, and wood ash for potassium",
        "summer_care": "Keep soil evenly moist. High heat can cause blossom drop; provide light afternoon shade if needed.",
        "monsoon_care": "Ensure prompt drainage. Excess moisture causes root suffocation and damping off.",
        "winter_care": "Moderate watering when the topsoil is dry. Keep in the sunniest corner of the garden.",
        "tips": "Spray cold-pressed neem oil solution weekly to protect young leaves against thrips and leaf-curl virus."
    },
    "Rose": {
        "scientific_name": "Rosa rubiginosa",
        "category": "Ornamental Flowering Shrub",
        "base_water_ml": 450,
        "ideal_sunlight": "Direct morning sunlight (5-6 hours)",
        "sunlight_hours": 6,
        "growth_stages": {
            "Pruned / New Shoots": (1, 20),
            "Bud Formation": (21, 45),
            "Full Bloom": (46, 80),
            "Dormancy / Pruning": (81, 150)
        },
        "soil": "Rich garden soil blended with 30% organic compost and coarse sand",
        "fertilizer": "Mustard cake liquid manure, bone meal, and Epsom salt (magnesium sulfate) monthly",
        "summer_care": "Water deeply in the early morning. Mulch the root zone with coco chips to keep roots cool.",
        "monsoon_care": "Black spot fungal disease is prevalent in rain. Apply copper fungicide paste to cut stem ends.",
        "winter_care": "Prime blooming season. Prune old dead wood and feed regularly with balanced rose fertilizer.",
        "tips": "Deadhead (cut off) faded blooms immediately to channel energy into vigorous new flower buds."
    },
    "Jasmine": {
        "scientific_name": "Jasminum sambac",
        "category": "Fragrant Flowering Shrub",
        "base_water_ml": 400,
        "ideal_sunlight": "Direct sunlight (6 hours)",
        "sunlight_hours": 6,
        "growth_stages": {
            "Vegetative Growth": (1, 30),
            "Bud Initiation": (31, 60),
            "Peak Bloom Period": (61, 120),
            "Post-Flowering": (121, 365)
        },
        "soil": "Porous, aerated sandy-clay soil with high organic humus",
        "fertilizer": "Cow manure powder, groundnut cake extract, and rock phosphate monthly",
        "summer_care": "Peak blooming phase requires regular daily morning watering. Never let roots dry out completely.",
        "monsoon_care": "Natural rainfall is sufficient. Inspect pot drainage holes to avoid standing water.",
        "winter_care": "Prune long woody branches in late winter to prepare for vibrant spring flowering.",
        "tips": "Prune lateral branch tips after each flowering cycle to trigger multiple new budding shoots."
    },
    "Snake Plant": {
        "scientific_name": "Sansevieria trifasciata",
        "category": "Indoor Air Purifier Succulent",
        "base_water_ml": 150,
        "ideal_sunlight": "Low to bright indirect light",
        "sunlight_hours": 2,
        "growth_stages": {
            "Young Pup": (1, 60),
            "Active Foliage Growth": (61, 180),
            "Established Mature Plant": (181, 999)
        },
        "soil": "Fast-draining cactus and succulent mix with pumice/perlite",
        "fertilizer": "Mild all-purpose liquid houseplant fertilizer once every 2-3 months",
        "summer_care": "Water once every 7 to 10 days. Ensure soil dries completely before the next watering.",
        "monsoon_care": "High humidity slows soil drying. Water only once every 14 to 20 days.",
        "winter_care": "Nearly dormant; water sparingly once every 3 to 4 weeks.",
        "tips": "Overwatering is the primary danger for Snake Plants. When in doubt, hold off on watering."
    },
    "Pothos (Money Plant)": {
        "scientific_name": "Epipremnum aureum",
        "category": "Indoor Air-Purifying Vine",
        "base_water_ml": 200,
        "ideal_sunlight": "Bright, indirect or filtered sunlight",
        "sunlight_hours": 3,
        "growth_stages": {
            "Stem Rooting": (1, 25),
            "Vine Elongation": (26, 90),
            "Dense Foliage Canopy": (91, 999)
        },
        "soil": "Blend of potting soil, cocopeat, and perlite in equal proportions",
        "fertilizer": "Diluted liquid seaweed extract or balanced houseplant fertilizer once a month",
        "summer_care": "Water twice a week when the top 1 inch of soil is dry. Mist leaves to enhance gloss.",
        "monsoon_care": "Retains moisture well; water every 5 to 7 days. Ensure good ambient airflow.",
        "winter_care": "Keep away from cold air-conditioner drafts and open chilly windows.",
        "tips": "Gently wipe leaves with a damp cloth monthly to clean dust and maintain photosynthesis efficiency."
    },
    "Aloe Vera": {
        "scientific_name": "Aloe barbadensis miller",
        "category": "Medicinal Succulent",
        "base_water_ml": 180,
        "ideal_sunlight": "Bright indirect light to gentle morning sun",
        "sunlight_hours": 4,
        "growth_stages": {
            "Pup / Offset": (1, 45),
            "Gel Layer Development": (46, 120),
            "Mature Harvestable": (121, 999)
        },
        "soil": "Sandy, porous, well-aerated succulent potting soil",
        "fertilizer": "Diluted organic compost tea once during spring or early summer",
        "summer_care": "Water once a week. If leaves turn reddish-brown, move the plant away from harsh direct midday sun.",
        "monsoon_care": "Shelter from heavy rain. Excess rainwater in the leaf rosettes will cause stem rot.",
        "winter_care": "Water minimally once every 2 to 3 weeks.",
        "tips": "Leaves store abundant gel and water; always allow soil to dry out between waterings."
    },
    "Monstera Deliciosa": {
        "scientific_name": "Monstera deliciosa",
        "category": "Tropical Indoor Foliage",
        "base_water_ml": 350,
        "ideal_sunlight": "Bright filtered or indirect sunlight",
        "sunlight_hours": 4,
        "growth_stages": {
            "Juvenile Solid Leaves": (1, 60),
            "Fenestration (Split Leaves)": (61, 180),
            "Mature Giant Vines": (181, 999)
        },
        "soil": "Chunky tropical aroid mix: bark, perlite, cocopeat, and compost",
        "fertilizer": "Balanced NPK (19:19:19) liquid fertilizer diluted monthly during the growing season",
        "summer_care": "Water once or twice a week when the top 2 inches of potting mix feel dry.",
        "monsoon_care": "Provide bright ambient light and good air circulation to prevent fungal leaf blight.",
        "winter_care": "Reduce watering frequency. Keep away from cold drafts and vents.",
        "tips": "Install a moss pole or coco coir support stake so aerial roots can climb and produce larger leaves."
    }
}

# Symptom Diagnostic Database
SYMPTOMS_DB = {
    "Yellowing Leaves": {
        "probable_cause": "Overwatering, Waterlogged Soil, or Nitrogen Deficiency",
        "remedy": "Check soil moisture. If soggy, hold off watering for 4-5 days and verify pot drainage holes. If soil is normal, supplement with organic compost or balanced nitrogen fertilizer."
    },
    "Brown Crispy Leaf Tips": {
        "probable_cause": "Underwatering, Low Ambient Humidity, or Sunburn",
        "remedy": "Water the plant thoroughly. Move away from harsh direct sun into bright indirect light. Mist leaves or use a humidity tray to increase ambient moisture."
    },
    "Drooping / Wilting Stems": {
        "probable_cause": "Severe Dehydration, Root Shock, or Advanced Root Rot",
        "remedy": "Inspect roots and soil. If bone dry, provide a deep bottom-watering soak. If soil is waterlogged and foul-smelling, repot immediately into fresh, well-draining dry soil."
    },
    "White Powdery Coating on Foliage": {
        "probable_cause": "Powdery Mildew (Fungal Infection)",
        "remedy": "Isolate the plant. Prune heavily affected leaves. Spray an organic solution of 1 tsp baking soda and 1/2 tsp mild liquid soap diluted in 1 liter of water, or use neem oil."
    },
    "Curled Stunted Leaves": {
        "probable_cause": "Thrips, Aphids, Spider Mites, or Leaf Curl Virus",
        "remedy": "Rinse the undersides of leaves with a firm stream of water. Spray cold-pressed neem oil combined with mild soap emulsion twice a week until clear."
    }
}
