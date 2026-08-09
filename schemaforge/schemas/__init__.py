"""Schema modules.

Importing this package imports every schema module, which registers each
``SPEC`` in the global registry at import time.
"""

from schemaforge.schemas import (  # noqa: F401
    contract,
    conversation,
    crm_record,
    email,
    form,
    insurance_claim,
    invoice,
    kg_triple,
    medical_note,
    receipt,
    resume,
    support_ticket,
)
