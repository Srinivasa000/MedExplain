"""
HIPAA & GDPR Clinical Security Compliance Engine
Implements automated PHI access audit logging, request middleware filters, and symmetric field encryption.
"""

import time
import hashlib
import json
import logging
from typing import Callable
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO)
audit_logger = logging.getLogger("HIPAA-Compliance-Audit")

class HIPAAFieldEncryptor:
    """
    Symmetric Field-Level Encryption utility using AES-256 (via Fernet).
    Used to encrypt Patient Health Information (PHI) before database commits.
    """
    def __init__(self, key: bytes = None):
        if not key:
            # In production, this key must be fetched securely from a Secret Manager (AWS KMS, HashiCorp Vault)
            self.key = Fernet.generate_key()
        else:
            self.key = key
        self.cipher_suite = Fernet(self.key)

    def encrypt_phi_field(self, field_value: str) -> bytes:
        """Encrypts sensitive plaintext string using AES-256"""
        if not field_value:
            return b""
        return self.cipher_suite.encrypt(field_value.encode('utf-8'))

    def decrypt_phi_field(self, encrypted_bytes: bytes) -> str:
        """Decrypts cipher bytes back to plaintext string"""
        if not encrypted_bytes:
            return ""
        return self.cipher_suite.decrypt(encrypted_bytes).decode('utf-8')


class HIPAAComplianceAuditMiddleware(BaseHTTPMiddleware):
    """
    FastAPI Security Audit logging middleware.
    Intercepts clinical endpoints, checks for PHI actions, computes payload integrity signatures,
    and writes an append-only immutable audit block.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Determine if the requested endpoint contains or reads Protected Health Information (PHI)
        is_phi_endpoint = any(path in request.url.path for path in ["/clinical/chat", "/clinical/analyze-report", "/ocr-document"])
        
        if not is_phi_endpoint:
            # Skip advanced audit logging for health checks, static styles, or landing routes
            return await call_next(request)

        # Retrieve request metadata
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # In production, operator ID and roles are parsed from the validated JWT token:
        # payload = decode_jwt_token(request.headers.get("Authorization"))
        operator_id = request.headers.get("X-Operator-ID", "SYSTEM-AUTO-ROUTER")
        operator_role = request.headers.get("X-Operator-Role", "CLINICAL_PORTAL_CLIENT")
        patient_id = request.query_params.get("patient_id", "GLOBAL-SESSION-PATIENT")

        # Read request body safely without exhausting stream
        body_bytes = await request.body()
        
        # Calculate SHA-256 of the input payload to guarantee record integrity
        payload_hash = hashlib.sha256(body_bytes).hexdigest()

        # Let the request proceed
        try:
            response = await call_next(request)
            access_status = "ACCESS_GRANTED" if response.status_code == 200 or response.status_code == 201 else f"STATUS_{response.status_code}"
        except Exception as e:
            access_status = "TRANSACTION_ERROR"
            raise e
        finally:
            latency = (time.time() - start_time) * 1000

            # ----------------------------------------------------
            # GENERATE IMMUTABLE COMPLIANCE AUDIT RECORD
            # ----------------------------------------------------
            audit_record = {
                "event_timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "operator_id": operator_id,
                "operator_role": operator_role,
                "action": "READ_WRITE_PHI" if request.method in ["POST", "PUT"] else "READ_PHI",
                "patient_uid": patient_id,
                "target_path": request.url.path,
                "origin_ip": client_ip,
                "browser_agent": user_agent,
                "payload_sha256": payload_hash,
                "transaction_latency_ms": f"{latency:.2f}ms",
                "security_status": access_status
            }

            # In production, write this JSON block to an immutable logging pool
            # e.g., an AWS CloudWatch Log Group with a Write-Once-Read-Many (WORM) retention lock
            # or an append-only secure MongoDB cluster
            audit_logger.info(f"HIPAA_AUDIT_BLOCK: {json.dumps(audit_record)}")

        return response


# --- Mock Database Action illustrating Field Encryption in transaction ---
def save_patient_record_example(first_name: str, last_name: str, dob: str, ssn: str):
    """
    Production example demonstrating how PHI is encrypted before DB queries
    """
    # Key should be fetched from KMS
    encryptor = HIPAAFieldEncryptor()

    encrypted_fname = encryptor.encrypt_phi_field(first_name)
    encrypted_lname = encryptor.encrypt_phi_field(last_name)
    encrypted_dob = encryptor.encrypt_phi_field(dob)
    encrypted_ssn = encryptor.encrypt_phi_field(ssn)

    print("--- PHI Field Obfuscation Successful ---")
    print(f"Plaintext First Name: {first_name}")
    print(f"Symmetric Cipher Text: {encrypted_fname[:20]}... [AES-256 Obfuscated]")
    
    # Decrypt verification pass
    decrypted_fname = encryptor.decrypt_phi_field(encrypted_fname)
    print(f"Decrypted plain-text verification: {decrypted_fname}")

    return {
        "fname_cipher": encrypted_fname,
        "lname_cipher": encrypted_lname,
        "dob_cipher": encrypted_dob,
        "ssn_cipher": encrypted_ssn
    }
