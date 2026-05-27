-- PostgreSQL Schema: HIPAA & GDPR Compliant Medical AI Data Infrastructure
-- Features: Affiliation mappings, multi-tenant schemas, record encryption, and append-only auditing.

-- Enable pgcrypto extension for database-level symmetric encryption of Protected Health Information (PHI)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Patients Table with AES-256 Symmetric Field-Level Encryption
CREATE TABLE patients (
    patient_id VARCHAR(64) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Encrypted PHI Fields (Stored as BYTEA to hold raw encrypted cipher text)
    -- Decrypted via: pgp_sym_decrypt(encrypted_first_name, 'SYMMETRIC_DECRYPTION_SECRET_KEY')
    encrypted_first_name BYTEA NOT NULL,
    encrypted_last_name BYTEA NOT NULL,
    encrypted_dob BYTEA NOT NULL,
    encrypted_email BYTEA NOT NULL,
    encrypted_phone BYTEA,
    encrypted_ssn BYTEA,
    
    -- Unencrypted clinical indexes (for search metrics without identifying individuals)
    gender VARCHAR(16),
    blood_type VARCHAR(8),
    tenant_id VARCHAR(64) NOT NULL
);

-- 2. Medical Reports Database (Multi-Tenant Isolation)
CREATE TABLE medical_reports (
    report_id VARCHAR(64) PRIMARY KEY,
    patient_id VARCHAR(64) REFERENCES patients(patient_id) ON DELETE CASCADE,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tenant_id VARCHAR(64) NOT NULL,
    
    -- Categorization & AI Analysis Outputs
    category VARCHAR(64) NOT NULL,               -- e.g. "BLOOD_TEST", "IMAGING"
    severity_level VARCHAR(16) NOT NULL,        -- e.g. "Low", "Moderate", "Critical"
    suggested_specialist VARCHAR(64),           -- e.g. "Cardiologist", "Endocrinologist"
    
    -- Document content
    encrypted_raw_text BYTEA NOT NULL,          -- AES encrypted OCR extracted clinical notes
    ai_structured_summary TEXT NOT NULL,         -- LLM synthesized clinical report summary
    
    -- Models auditing metadata
    ocr_engine_used VARCHAR(64),                -- e.g. "Google-DocumentAI-Medical"
    nlp_model_used VARCHAR(64),                 -- e.g. "PubMedBERT-Clinical-NER"
    reasoning_model_used VARCHAR(64)            -- e.g. "Med-PaLM 2 / Meditron-70B"
);

-- 3. Physiological Report Anomalies Table
CREATE TABLE report_anomalies (
    anomaly_id BIGSERIAL PRIMARY KEY,
    report_id VARCHAR(64) REFERENCES medical_reports(report_id) ON DELETE CASCADE,
    parameter_code VARCHAR(32) NOT NULL,       -- e.g., "GLUCOSE", "WBC"
    parameter_name VARCHAR(128) NOT NULL,      -- e.g., "Fasting Glucose", "White Blood Cells"
    observed_value NUMERIC(10, 2) NOT NULL,
    unit VARCHAR(16) NOT NULL,                 -- e.g., "mg/dL", "K/uL"
    status VARCHAR(16) NOT NULL,               -- e.g., "HIGH", "LOW", "NORMAL"
    clinical_significance TEXT
);

-- 4. Prescription Auditing & Scheduling Database
CREATE TABLE prescriptions (
    prescription_id VARCHAR(64) PRIMARY KEY,
    patient_id VARCHAR(64) REFERENCES patients(patient_id) ON DELETE CASCADE,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tenant_id VARCHAR(64) NOT NULL,
    
    -- Safe interaction flags
    drug_interaction_risk VARCHAR(16) NOT NULL,  -- e.g. "Low", "Moderate", "High"
    interaction_warning_details TEXT,            -- Description of drug-drug interactions
    
    -- Model audit
    ner_model_used VARCHAR(64)
);

-- 5. Individual Medication Items inside Prescription (One-to-Many relationship)
CREATE TABLE prescription_items (
    item_id BIGSERIAL PRIMARY KEY,
    prescription_id VARCHAR(64) REFERENCES prescriptions(prescription_id) ON DELETE CASCADE,
    medication_name VARCHAR(128) NOT NULL,      -- e.g., "Metformin", "Aspirin"
    dosage VARCHAR(64) NOT NULL,                 -- e.g., "500mg"
    timing_protocol VARCHAR(256) NOT NULL,       -- e.g., "twice daily with meals"
    indicated_purpose TEXT,                      -- e.g., "Type-2 Diabetes management"
    side_effects TEXT[]                          -- Array of extracted clinical side effects
);

-- 6. Immutable HIPAA Access & Change Audit Trail Table (Strict compliance)
-- This table is strictly INSERT-only. No UPDATE or DELETE privileges are granted.
CREATE TABLE hipaa_audit_trail (
    audit_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    operator_id VARCHAR(64) NOT NULL,           -- Practitioner, Patient, or Automated System UID
    operator_role VARCHAR(32) NOT NULL,         -- e.g. "PHYSICIAN", "PATIENT", "ADMIN", "AI_AGENT"
    action_type VARCHAR(16) NOT NULL,           -- e.g. "READ_PHI", "WRITE_PHI", "EXPORT_PHI", "LOGIN"
    target_patient_id VARCHAR(64) NOT NULL,     -- The patient ID whose records were accessed
    target_record_id VARCHAR(64),               -- Specific report_id or prescription_id
    
    -- Operational Security metadata
    client_ip VARCHAR(45) NOT NULL,             -- Supports IPv4 & IPv6
    user_agent TEXT,
    payload_sha256 CHAR(64) NOT NULL,           -- SHA-256 hash of payload to guarantee integrity
    access_status VARCHAR(16) NOT NULL          -- e.g. "GRANTED", "DENIED"
);

-- Restrict mutations on the audit trail table to guarantee immutability
REVOKE UPDATE, DELETE ON hipaa_audit_trail FROM PUBLIC;

-- ----------------------------------------------------
-- MONGO DB DOCUMENT SCHEMA REFERENCE (REPORTS MAPPING)
-- ----------------------------------------------------
/*
{
  "_id": "DOC-9921021",
  "patient_id": "PAT-883921",
  "upload_date": "2026-05-27T14:35:57Z",
  "tenant_id": "CLINIC-EAST-88",
  "clinical_records": {
    "document_type": "laboratory_sheet",
    "extracted_raw_text": "LAB REPORT: Glucose 185 mg/dL...",
    "category": "CHEMISTRY",
    "severity": "Moderate",
    "physician_specialty_suggested": "Endocrinologist",
    "anomalies": [
      {
        "parameter": "GLUCOSE",
        "name": "Fasting Glucose",
        "value": 185,
        "unit": "mg/dL",
        "status": "HIGH",
        "clinical_significance": "Suggests acute hyperglycemia, requiring clinical assessment."
      }
    ],
    "recommendations": [
      "Retest plasma fasting glucose",
      "Adopt a low glycemic index nutritional routine"
    ],
    "ai_engine_metadata": {
      "ocr_system": "Google Document AI v1.2",
      "ner_tagger": "PubMedBERT-Clinical-NER",
      "reasoning_agent": "Meditron-70B vLLM pipeline"
    }
  }
}
*/
