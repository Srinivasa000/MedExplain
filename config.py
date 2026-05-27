"""
Configuration file for Automated Clinical AI Assistant Platform
Features medically trained models configurations, report reference ranges, and Rx safety mappings.
"""

class Config:
    # --- Medically Trained AI Models Configuration ---
    CLINICAL_NLP_MODEL = "PubMedBERT-Clinical-NER"      # Medically trained biomedical encoder
    CLINICAL_REASONING_MODEL = "Med-PaLM 2 / Meditron-70B" # SOTA Clinical LLMs
    RADIOLOGY_MODEL = "CheXNet (121-layer DenseNet)"    # Trained on 112,000+ chest X-rays
    MULTIMODAL_ALIGNMENT_MODEL = "BioViL"               # Joint chest X-ray & note encoder
    EMBEDDING_MODEL = "BioClinical-PubMedBERT-Embeddings" # Medically trained embedding indexer

    # Model hyperparameters
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 50

    # Data paths
    TRAIN_DATA_PATH = 'data/train.txt'
    TEST_DATA_PATH = 'data/test.txt'
    MODEL_SAVE_PATH = 'models/drug_ner_model.pth'
    VOCAB_PATH = 'models/vocab.pkl'

    # Entity tags (BIO format for PubMedBERT Clinical NER)
    ENTITY_TAGS = {
        'O': 0,           # Outside
        'B-DRUG': 1,      # Begin Drug Name
        'I-DRUG': 2,      # Inside Drug Name
        'B-DOSAGE': 3,    # Begin Dosage
        'I-DOSAGE': 4,    # Inside Dosage
        'B-EFFECT': 5,    # Begin Drug Effect
        'I-EFFECT': 6,    # Inside Drug Effect
    }

    TAG_TO_IDX = ENTITY_TAGS
    IDX_TO_TAG = {v: k for k, v in ENTITY_TAGS.items()}
    NUM_TAGS = len(ENTITY_TAGS)

    # Special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'

    # Lab Report Classification (Medically Standard Categories)
    LAB_CATEGORIES = {
        'BLOOD_TEST': 0,        # Complete Blood Count, Basic Metabolic Panel
        'URINE_TEST': 1,        # Urinalysis, urine culture
        'CARDIAC_TEST': 2,      # EKG, Echo, Stress test
        'IMAGING': 3,           # X-ray, CT, MRI, Ultrasound
        'PATHOLOGY': 4,         # Biopsy, tissue analysis
        'MICROBIOLOGY': 5,      # Culture, sensitivity testing
        'CHEMISTRY': 6,         # Liver function, kidney function
        'IMMUNOLOGY': 7,        # Allergy tests, autoimmune tests
        'HORMONE_TEST': 8,      # Thyroid, diabetes, hormone panels
        'OTHER': 9              # Miscellaneous tests
    }

    LAB_CATEGORY_TO_IDX = LAB_CATEGORIES
    IDX_TO_LAB_CATEGORY = {v: k for k, v in LAB_CATEGORIES.items()}
    NUM_LAB_CATEGORIES = len(LAB_CATEGORIES)

    # Lab report model paths
    LAB_MODEL_SAVE_PATH = 'models/lab_classifier_model.pth'
    LAB_VOCAB_PATH = 'models/lab_vocab.pkl'
    
    # Dosage number classification regex
    DOSAGE_PATTERNS = {
        'MG': r'\d+\s*mg\b',
        'MCG': r'\d+\s*mcg\b',
        'ML': r'\d+\s*ml\b',
        'G': r'\d+\s*g\b',
        'IU': r'\d+\s*iu\b',
        'TABLETS': r'\d+\s*tablets?\b',
        'CAPSULES': r'\d+\s*capsules?\b',
        'DROPS': r'\d+\s*drops?\b',
        'SPRAYS': r'\d+\s*sprays?\b',
        'PATCHES': r'\d+\s*patches?\b',
    }

    # --- Medical Guidelines Reference Ranges (RAG Base) ---
    LAB_REFERENCE_RANGES = {
        'WBC': {'min': 4.5, 'max': 11.0, 'unit': 'K/uL', 'name': 'White Blood Cells'},
        'RBC': {'min': 4.3, 'max': 5.9, 'unit': 'M/uL', 'name': 'Red Blood Cells'},
        'HGB': {'min': 13.5, 'max': 17.5, 'unit': 'g/dL', 'name': 'Hemoglobin'},
        'HCT': {'min': 41.0, 'max': 50.0, 'unit': '%', 'name': 'Hematocrit'},
        'PLATELETS': {'min': 150, 'max': 450, 'unit': 'K/uL', 'name': 'Platelets'},
        'GLUCOSE': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'name': 'Fasting Glucose'},
        'BUN': {'min': 7, 'max': 20, 'unit': 'mg/dL', 'name': 'Blood Urea Nitrogen'},
        'CREATININE': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL', 'name': 'Creatinine'},
        'SODIUM': {'min': 136, 'max': 145, 'unit': 'mEq/L', 'name': 'Sodium'},
        'POTASSIUM': {'min': 3.5, 'max': 5.1, 'unit': 'mEq/L', 'name': 'Potassium'},
        'CHOLESTEROL': {'min': 100, 'max': 200, 'unit': 'mg/dL', 'name': 'Total Cholesterol'},
        'TRIGLYCERIDES': {'min': 10, 'max': 150, 'unit': 'mg/dL', 'name': 'Triglycerides'},
        'ALT': {'min': 7, 'max': 56, 'unit': 'U/L', 'name': 'Alanine Aminotransferase'},
        'AST': {'min': 10, 'max': 40, 'unit': 'U/L', 'name': 'Aspartate Aminotransferase'},
        'TSH': {'min': 0.4, 'max': 4.0, 'unit': 'uIU/mL', 'name': 'Thyroid Stimulating Hormone'},
    }

    # --- Clinical Pharmacology Database (Layperson-Friendly Descriptions) ---
    CLINICAL_DRUG_DB = {
        'aspirin': {
            'purpose': 'A common medicine used to help prevent blood clots, protect the heart, and relieve minor pain.',
            'side_effects': ['Stomach upset or heartburn', 'Easy bruising', 'Ringing in the ears (tinnitus)', 'Stomach bleeding (rare)'],
            'interactions': ['Warfarin', 'Ibuprofen', 'Apixaban', 'Clopidogrel'],
            'timing': 'Take once daily in the morning with a full glass of water, ideally with food to protect your stomach.'
        },
        'metformin': {
            'purpose': 'A widely used daily medicine that helps lower and control high blood sugar in patients with Type-2 diabetes.',
            'side_effects': ['Diarrhea', 'Feeling sick to your stomach (nausea)', 'Bloating or gas', 'Low Vitamin B12 levels over time'],
            'interactions': ['Contrast dyes (iodinated)', 'Cimetidine', 'Topiramate'],
            'timing': 'Take twice daily with breakfast and dinner to help prevent stomach upset.'
        },
        'ibuprofen': {
            'purpose': 'A common pain reliever and fever reducer that also helps lower swelling, joint stiffness, and inflammation.',
            'side_effects': ['Stomach pain or heartburn', 'Fluid retention or bloating', 'Mild increase in blood pressure'],
            'interactions': ['Aspirin', 'Lisinopril', 'Furosemide', 'Warfarin'],
            'timing': 'Take every 6 to 8 hours as needed for pain, always with food or milk.'
        },
        'omeprazole': {
            'purpose': 'A medicine that reduces acid production in your stomach to treat frequent heartburn, indigestion, and acid reflux.',
            'side_effects': ['Headache', 'Nausea or gas', 'Mild diarrhea'],
            'interactions': ['Clopidogrel', 'Ketoconazole', 'Atazanavir'],
            'timing': 'Take once daily 30 to 60 minutes before your first meal of the day.'
        },
        'lisinopril': {
            'purpose': 'A daily blood pressure medicine used to lower high blood pressure and help protect the heart after a cardiac event.',
            'side_effects': ['Dry, tickling cough', 'Dizziness or lightheadedness when standing', 'High potassium levels in the blood'],
            'interactions': ['Spironolactone', 'Potassium supplements', 'Ibuprofen'],
            'timing': 'Take once daily at approximately the same time, with or without food.'
        },
        'atorvastatin': {
            'purpose': 'A daily cholesterol-lowering pill used to keep blood vessels healthy, clear, and reduce the risk of heart attacks.',
            'side_effects': ['Mild muscle pain or stiffness', 'Slight increase in blood sugar', 'Tiredness'],
            'interactions': ['Clarithromycin', 'Gemfibrozil', 'Grapefruit juice'],
            'timing': 'Take once daily in the evening or at bedtime, as the liver processes cholesterol mostly at night.'
        },
        'amoxicillin': {
            'purpose': 'A common penicillin-class antibiotic medicine used to fight and cure bacterial infections (like throat or ear infections).',
            'side_effects': ['Skin rash', 'Stomach upset or nausea', 'Loose stools (diarrhea)', 'Yeast infections (thrush)'],
            'interactions': ['Oral contraceptives', 'Allopurinol', 'Methotrexate'],
            'timing': 'Take every 8 hours (three times daily) for the full prescribed duration, even if you feel better early.'
        },
        'sertraline': {
            'purpose': 'A daily medicine used to help manage depression, panic, and persistent anxiety, improving general mood and energy.',
            'side_effects': ['Nausea or dry mouth', 'Trouble sleeping (insomnia)', 'Sweating more than usual', 'Low sex drive'],
            'interactions': ['MAOIs', 'Tramadol', 'St. John\'s wort', 'NSAIDs'],
            'timing': 'Take once daily in the morning or evening, consistently either with food or without food.'
        }
    }

    # --- Emergency Escalation Symptom Keywords ---
    EMERGENCY_SYMPTOMS = [
        "chest pain", "crushing pressure", "shortness of breath", "difficulty breathing",
        "stroke", "facial droop", "arm weakness", "slurred speech", "loss of consciousness",
        "severe allergic reaction", "anaphylaxis", "throat swelling", "suicidal thoughts", "severe head injury"
    ]

    # --- Medical Disclaimer Footer Text ---
    DISCLAIMER_TEXT = (
        "IMPORTANT MEDICAL DISCLAIMER: This platform is a clinical decision-support tool powered by medically trained "
        "AI models including PubMedBERT, BioViL, and Meditron-70B. It is provided for informational and educational "
        "purposes only and does NOT constitute medical advice. It does not replace a professional clinical diagnosis, "
        "treatment plan, or emergency healthcare intervention. Always consult a qualified physician or clinical professional "
        "for medical concerns. If you are experiencing a life-threatening emergency, call 911 or visit the nearest ER immediately."
    )

    # Evaluation
    RANDOM_SEED = 42
