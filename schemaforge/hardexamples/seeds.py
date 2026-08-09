"""Clean seed generators for hard-example generation.

``build_seed(schema_name, rng)`` returns a ``(document_text, gold)`` pair for
any registered schema: a clean document and the gold JSON that the schema's
pydantic model validates against.  Every gold string value literally appears in
the clean document text (before any corruption); that is what makes the
hallucination support-check in the evaluation harness meaningful.

All content is synthesized from small hardcoded vocabularies plus the seeded
``rng``.  There is no network access, no LLM call, and no external data.
"""

from __future__ import annotations

import datetime as _dt
import random
from decimal import Decimal
from typing import Callable

from schemaforge.registry import get_schema

_FIRST = ("Jane", "John", "Maria", "David", "Aisha", "Robert", "Elena", "Michael", "Priya", "Tomas")
_LAST = ("Doe", "Smith", "Garcia", "Chen", "Patel", "Kim", "Brown", "Nguyen", "Okafor", "Larsen")

_DESCRIPTIONS = (
    "Industrial Widget",
    "Precision Gadget",
    "Rubber Hose",
    "Flux Capacitor Unit",
    "Copper Wire Coil",
    "Backup Battery Pack",
)
_GROCERIES = (
    "Organic Bananas",
    "Whole Milk",
    "Sourdough Bread",
    "Free-Range Eggs",
    "Ground Coffee",
    "Almond Butter",
)
_UNIT_PRICES = ("12.50", "99.99", "450.00", "7.25", "1500.00", "3.99")


def _pick(rng: random.Random, options: tuple) -> str:
    """Return one element of ``options`` chosen by ``rng``."""
    return options[rng.randrange(len(options))]


def _person(rng: random.Random) -> str:
    """Return a ``"First Last"`` name from the small name vocabularies."""
    return f"{_pick(rng, _FIRST)} {_pick(rng, _LAST)}"


def _date(rng: random.Random) -> _dt.date:
    """Return a random date; days stay within 1..28 so every month is valid."""
    return _dt.date(rng.choice((2025, 2026)), rng.randrange(1, 13), rng.randrange(1, 29))


def _date_plus(day: _dt.date, delta: int) -> _dt.date:
    """Return ``day`` plus ``delta`` calendar days."""
    return day + _dt.timedelta(days=delta)


def _medical_note(rng: random.Random) -> tuple[str, dict]:
    patient = _person(rng)
    age = rng.randrange(24, 88)
    visit = _date(rng)
    follow_up = _date_plus(visit, rng.randrange(21, 45))
    diagnosis = _pick(
        rng,
        (
            "Hypertension",
            "Type 2 Diabetes Mellitus",
            "Myocardial Infarction",
            "Chronic Obstructive Pulmonary Disease",
            "Chronic Kidney Disease",
            "Congestive Heart Failure",
        ),
    )
    physician = _pick(
        rng,
        ("Dr. Smith", "Dr. Lee", "Dr. Patel", "Dr. Nguyen", "Dr. Okafor", "Dr. Rossi"),
    )
    med_name, med_dosage = _pick(
        rng,
        (
            ("lisinopril", "10 mg"),
            ("metformin", "500 mg"),
            ("atorvastatin", "20 mg"),
            ("amlodipine", "5 mg"),
            ("omeprazole", "20 mg"),
            ("metoprolol", "25 mg"),
        ),
    )
    text = (
        f"{patient}, age {age}, seen {visit.isoformat()} for {diagnosis}. "
        f"Medications: {med_name} {med_dosage}. Attending: {physician}. "
        f"Follow-up {follow_up.isoformat()}."
    )
    gold = {
        "patient_name": patient,
        "patient_age": age,
        "visit_date": visit.isoformat(),
        "diagnosis": diagnosis,
        "physician_name": physician,
        "medications": [{"name": med_name, "dosage": med_dosage}],
        "follow_up_date": follow_up.isoformat(),
    }
    return text, gold


def _invoice(rng: random.Random) -> tuple[str, dict]:
    vendor, vendor_email, vendor_phone = _pick(
        rng,
        (
            ("Acme Corporation", "billing@acme.com", "+1 555 0100"),
            ("Globex Inc.", "accounts@globex.com", "+1 555 0123"),
            ("Initech LLC", "payables@initech.com", "+1 555 0145"),
            ("Umbrella Corp", "finance@umbrella.com", "+1 555 0167"),
        ),
    )
    invoice_number = f"INV-{rng.randrange(1000, 9999)}"
    invoice_date = _date(rng)
    due_date = _date_plus(invoice_date, 30)

    line_items: list[dict] = []
    item_lines: list[str] = []
    for _ in range(rng.randrange(1, 4)):
        description = _pick(rng, _DESCRIPTIONS)
        quantity = rng.randrange(1, 6)
        unit_price = _pick(rng, _UNIT_PRICES)
        line_total = (Decimal(quantity) * Decimal(unit_price)).quantize(Decimal("0.01"))
        line_items.append(
            {
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": format(line_total, "f"),
            }
        )
        item_lines.append(f"{description} x{quantity} at {unit_price} each (line {line_total})")

    subtotal = (
        sum(
            (Decimal(item["quantity"]) * Decimal(item["unit_price"]) for item in line_items),
            Decimal("0"),
        )
    ).quantize(Decimal("0.01"))
    tax = (subtotal * Decimal("0.08")).quantize(Decimal("0.01"))
    total = (subtotal + tax).quantize(Decimal("0.01"))

    text = (
        f"Invoice {invoice_number} dated {invoice_date.isoformat()}, due {due_date.isoformat()}, "
        f"from {vendor} ({vendor_email}, {vendor_phone}). "
        f"Items: {'; '.join(item_lines)}. "
        f"Subtotal: {format(subtotal, 'f')}, Tax: {format(tax, 'f')}, "
        f"Total: {format(total, 'f')}."
    )
    gold = {
        "invoice_number": invoice_number,
        "invoice_date": invoice_date.isoformat(),
        "due_date": due_date.isoformat(),
        "vendor_name": vendor,
        "vendor_email": vendor_email,
        "vendor_phone": vendor_phone,
        "total_amount": format(total, "f"),
        "tax_amount": format(tax, "f"),
        "line_items": line_items,
    }
    return text, gold


def _receipt(rng: random.Random) -> tuple[str, dict]:
    store = _pick(rng, ("Green Grocers", "CityMart", "Terra Market", "Quick Stop", "Fresh Basket"))
    cashier = _pick(rng, ("Tara", "Miguel", "Priya", "Omar", "Lena"))
    payment = _pick(rng, ("Credit Card", "Cash", "Debit Card", "Mobile Pay"))
    receipt_number = f"RC-{rng.randrange(10000, 99999)}"
    receipt_date = _date(rng)
    refunded = rng.random() < 0.15

    line_items: list[dict] = []
    item_lines: list[str] = []
    for _ in range(rng.randrange(1, 4)):
        description = _pick(rng, _GROCERIES)
        quantity = rng.randrange(1, 5)
        unit_price = _pick(rng, _UNIT_PRICES)
        line_total = (Decimal(quantity) * Decimal(unit_price)).quantize(Decimal("0.01"))
        line_items.append(
            {
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": format(line_total, "f"),
            }
        )
        item_lines.append(f"{description} x{quantity} at {unit_price} (line {line_total})")

    subtotal = (
        sum(
            (Decimal(item["quantity"]) * Decimal(item["unit_price"]) for item in line_items),
            Decimal("0"),
        )
    ).quantize(Decimal("0.01"))
    tax = (subtotal * Decimal("0.08")).quantize(Decimal("0.01"))
    total = (subtotal + tax).quantize(Decimal("0.01"))

    refund_note = " This receipt was refunded." if refunded else ""
    text = (
        f"Receipt {receipt_number} — {store}, {receipt_date.isoformat()}. "
        f"Cashier: {cashier}. Items: {'; '.join(item_lines)}. "
        f"Subtotal: {format(subtotal, 'f')}, Tax: {format(tax, 'f')}, "
        f"Total: {format(total, 'f')}. Paid by: {payment}.{refund_note}"
    )
    gold = {
        "receipt_number": receipt_number,
        "receipt_date": receipt_date.isoformat(),
        "store_name": store,
        "cashier_name": cashier,
        "total_amount": format(total, "f"),
        "tax_amount": format(tax, "f"),
        "payment_method": payment,
        "refunded": refunded,
        "line_items": line_items,
    }
    return text, gold


def _resume(rng: random.Random) -> tuple[str, dict]:
    first = _pick(rng, _FIRST)
    last = _pick(rng, _LAST)
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}@example.com"
    phone = f"+1 555 {rng.randrange(1000, 9999)}"
    title = _pick(
        rng,
        ("Senior Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer", "Frontend Developer"),
    )
    years = float(rng.randrange(1, 16))
    degree = _pick(rng, ("Bachelor of Science", "Master of Science", "Ph.D.", "Bachelor of Arts"))
    employer = _pick(
        rng,
        ("Northern Lights Labs", "Bluebird Software", "Keystone Analytics", "Copperfield Systems"),
    )
    location = _pick(rng, ("Austin, TX", "Seattle, WA", "New York, NY", "Denver, CO"))

    text = (
        f"{name} — {title}. Email: {email}, phone {phone}. "
        f"{years:g} years of experience. Highest degree: {degree}. "
        f"Current employer: {employer}. Location: {location}."
    )
    gold = {
        "candidate_name": name,
        "email": email,
        "phone": phone,
        "job_title": title,
        "years_experience": years,
        "highest_degree": degree,
        "current_employer": employer,
        "location": location,
    }
    return text, gold


def _contract(rng: random.Random) -> tuple[str, dict]:
    ctype = _pick(
        rng,
        (
            "Software License Agreement",
            "Service Agreement",
            "Non-Disclosure Agreement",
            "Consulting Agreement",
            "Master Services Agreement",
        ),
    )
    party_a = _pick(rng, ("Acme Corporation", "Northwind Systems", "Vertex Industries", "Aurora Holdings"))
    party_b = _pick(rng, ("Beta Labs", "Crestline Solutions", "Delta Robotics", "Osprey Media"))
    start = _date(rng)
    end = _date_plus(start, rng.randrange(90, 365))
    auto_renewal = rng.random() < 0.4
    cap = f"{rng.randrange(1, 10)}00000.00"
    jurisdiction = _pick(rng, ("State of Delaware", "State of California", "State of New York", "State of Texas"))

    renewal = " This agreement renews automatically." if auto_renewal else ""
    text = (
        f"{ctype} between {party_a} and {party_b}, effective {start.isoformat()}, "
        f"expiring {end.isoformat()}. Liability cap: {cap}.{renewal} "
        f"Governing law: {jurisdiction}."
    )
    gold = {
        "contract_type": ctype,
        "party_a": party_a,
        "party_b": party_b,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "auto_renewal": auto_renewal,
        "liability_cap": cap,
        "jurisdiction": jurisdiction,
    }
    return text, gold


def _support_ticket(rng: random.Random) -> tuple[str, dict]:
    ticket_id = f"TKT-{rng.randrange(10000, 99999)}"
    created = _date(rng)
    intent = _pick(rng, ("Refund", "Cancellation", "Battery", "Display", "Billing Inquiry", "Product Setup"))
    category = _pick(rng, ("Billing", "Hardware", "Software", "Delivery"))
    product = _pick(rng, ("Phone X", "Laptop Pro", "Tablet Mini", "Headphones Plus", "Charger Dock"))
    urgency = _pick(rng, ("High", "Medium", "Low"))
    sentiment = _pick(rng, ("Frustrated", "Neutral", "Satisfied", "Angry"))
    resolution = _pick(rng, ("Resolved", "Escalated", "Pending", "Closed"))
    customer_email = f"{_pick(rng, _FIRST).lower()}@customer.net"

    text = (
        f"Ticket {ticket_id} created {created.isoformat()}. Customer {customer_email} requests "
        f"a {intent} for the {product}. Category: {category}. Urgency: {urgency}. "
        f"Sentiment: {sentiment}. Status: {resolution}."
    )
    gold = {
        "ticket_id": ticket_id,
        "created_date": created.isoformat(),
        "intent": intent,
        "issue_category": category,
        "product": product,
        "urgency": urgency,
        "sentiment": sentiment,
        "resolution_status": resolution,
        "customer_email": customer_email,
    }
    return text, gold


def _insurance_claim(rng: random.Random) -> tuple[str, dict]:
    claim_number = f"CLM-{rng.randrange(10000, 99999)}"
    policy_number = f"POL-{rng.randrange(1000, 9999)}"
    claimant = _person(rng)
    claim_type = _pick(rng, ("Auto Collision", "Fire", "Water Damage", "Theft", "Liability"))
    incident = _date(rng)
    amount = f"{rng.randrange(1, 30)}000.00"
    status = _pick(rng, ("Open", "Under Review", "Approved", "Denied"))
    party2 = _person(rng)
    role2 = _pick(rng, ("Witness", "Other Driver", "Passenger"))

    text = (
        f"Claim {claim_number}, policy {policy_number}. Claimant: {claimant}. "
        f"The incident involved {claim_type} on {incident.isoformat()}. "
        f"Amount requested: {amount}. Status: {status}. "
        f"Other party: {party2} ({role2})."
    )
    gold = {
        "claim_number": claim_number,
        "policy_number": policy_number,
        "claimant_name": claimant,
        "claim_type": claim_type,
        "incident_date": incident.isoformat(),
        "amount_requested": amount,
        "status": status,
        "involved_parties": [{"name": party2, "role": role2}],
    }
    return text, gold


def _crm_record(rng: random.Random) -> tuple[str, dict]:
    contact = _person(rng)
    company = _pick(rng, ("Acme Corporation", "Northwind Systems", "Vertex Industries", "Aurora Holdings"))
    email = f"{contact.split()[0].lower()}@{company.split()[0].lower()}.com"
    phone = f"+1 555 {rng.randrange(1000, 9999)}"
    stage = _pick(rng, ("Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"))
    value = f"{rng.randrange(1, 20)}000.00"
    probability = rng.choice((0.2, 0.4, 0.6, 0.8))
    owner = _pick(rng, ("Sarah", "Kofi", "Rita", "Devan"))
    last_contact = _date(rng)

    text = (
        f"{contact} at {company} ({email}, {phone}). Deal stage: {stage}, "
        f"value {value}, probability {probability}. Owner: {owner}. "
        f"Last contact: {last_contact.isoformat()}."
    )
    gold = {
        "contact_name": contact,
        "company_name": company,
        "email": email,
        "phone": phone,
        "deal_stage": stage,
        "deal_value": value,
        "probability": probability,
        "owner": owner,
        "last_contact_date": last_contact.isoformat(),
    }
    return text, gold


def _email(rng: random.Random) -> tuple[str, dict]:
    first1 = _pick(rng, _FIRST)
    first2 = _pick(rng, _FIRST)
    while first2 == first1:
        first2 = _pick(rng, _FIRST)
    sender = f"{first1.lower()}@workmail.com"
    recipient = f"{first2.lower()}@workmail.com"
    subject = _pick(
        rng,
        ("Quarterly invoice attached", "Meeting rescheduled", "New login credentials", "Vacation request approval"),
    )
    body = _pick(
        rng,
        (
            "Please review the attached invoice for Q3 and let me know if anything is missing.",
            "The team meeting moved to Thursday at ten in the morning.",
            "Your new account credentials are attached; please change the password on first login.",
            "Your vacation request for next month has been approved.",
        ),
    )
    sent = _date(rng)
    importance = _pick(rng, ("High", "Normal", "Low"))
    attachments = rng.random() < 0.6
    reply = rng.random() < 0.5

    text = (
        f"From: {sender}\nTo: {recipient}\nSubject: {subject}\n"
        f"Sent: {sent.isoformat()} ({importance} importance)\n"
        f"Attachment: {'yes' if attachments else 'no'}\n"
        f"Reply needed: {'yes' if reply else 'no'}\n\n{body}"
    )
    gold = {
        "sender_email": sender,
        "recipient_email": recipient,
        "subject": subject,
        "body": body,
        "sent_date": sent.isoformat(),
        "importance": importance,
        "attachments_present": attachments,
        "reply_needed": reply,
    }
    return text, gold


def _conversation(rng: random.Random) -> tuple[str, dict]:
    channel = _pick(rng, ("Chat", "Email", "Phone"))
    started = _date(rng)
    customer = _person(rng)
    resolved = rng.random() < 0.5

    turn_texts = (
        "Hi, I need help with my order.",
        "Can you tell me the status?",
        "Thanks, that helped.",
        "My account still shows the old address.",
        "I will try that now.",
        "When will it be shipped?",
        "The login page keeps failing.",
        "Please escalate this to a specialist.",
    )
    turns: list[dict] = []
    lines: list[str] = []
    for i in range(rng.randrange(3, 6)):
        speaker = "Customer" if i % 2 == 0 else "Support"
        turn_text = _pick(rng, turn_texts)
        timestamp = f"{rng.randrange(9, 18):02d}:{rng.randrange(0, 60):02d}"
        turns.append({"speaker": speaker, "text": turn_text, "timestamp": timestamp})
        lines.append(f"{speaker}: {turn_text} ({timestamp})")

    resolution = " Issue resolved: yes." if resolved else " Issue resolved: no."
    text = (
        f"Conversation on {channel}, started {started.isoformat()}. "
        f"Customer: {customer}.{resolution}\n" + "\n".join(lines)
    )
    gold = {
        "channel": channel,
        "started_date": started.isoformat(),
        "customer_name": customer,
        "resolved": resolved,
        "turns": turns,
    }
    return text, gold


def _form(rng: random.Random) -> tuple[str, dict]:
    form_id = f"FRM-{rng.randrange(10000, 99999)}"
    submitted = _date(rng)
    submitter = _person(rng)
    submitter_email = f"{submitter.split()[0].lower()}.{submitter.split()[1].lower()}@corp.example"
    request_type = _pick(rng, ("New Account", "Access Request", "Bug Report", "Feature Request", "Data Deletion"))
    details = _pick(
        rng,
        (
            "I need read-only access to the reporting dashboard.",
            "The export button returns a blank file on the staging server.",
            "Please add a dark mode option to the settings page.",
            "Delete all records tied to the terminated employee.",
        ),
    )
    status = _pick(rng, ("Submitted", "In Review", "Approved", "Rejected"))
    priority = _pick(rng, ("Low", "Medium", "High"))

    text = (
        f"Form {form_id} submitted {submitted.isoformat()} by {submitter} ({submitter_email}). "
        f"Request type: {request_type}. Details: {details}. Status: {status}. "
        f"Priority: {priority}."
    )
    gold = {
        "form_id": form_id,
        "submitted_date": submitted.isoformat(),
        "submitter_name": submitter,
        "submitter_email": submitter_email,
        "request_type": request_type,
        "details": details,
        "status": status,
        "priority": priority,
    }
    return text, gold


def _kg_triple(rng: random.Random) -> tuple[str, dict]:
    subject, predicate, obj = _pick(
        rng,
        (
            ("Acme Corporation", "acquired", "Beta Labs"),
            ("Northwind Systems", "partners with", "Delta Robotics"),
            ("Vertex Industries", "supplies", "Osprey Media"),
            ("Aurora Holdings", "invests in", "Crestline Solutions"),
        ),
    )
    subject_type = _pick(rng, ("Organization", "Product", "Person", "Technology"))
    object_type = _pick(rng, ("Organization", "Product", "Person", "Technology"))
    context = _pick(
        rng,
        (
            "This relationship was stated in the annual report filed in March.",
            "The acquisition was announced during the quarterly earnings call.",
            "The partnership is documented in the supplier contract from last year.",
        ),
    )
    confidence = rng.choice((0.7, 0.8, 0.9, 0.95))
    extracted = _date(rng)

    text = (
        f"Triple extracted: {extracted.isoformat()}. {subject} --{predicate}--> {obj}. "
        f"{subject} is an {subject_type}; {obj} is an {object_type}. "
        f"Context: {context}. Confidence: {confidence}."
    )
    gold = {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "subject_type": subject_type,
        "object_type": object_type,
        "context": context,
        "confidence": confidence,
        "extracted_date": extracted.isoformat(),
    }
    return text, gold


_BUILDERS: dict[str, Callable[[random.Random], tuple[str, dict]]] = {
    "medical_note": _medical_note,
    "invoice": _invoice,
    "receipt": _receipt,
    "resume": _resume,
    "contract": _contract,
    "support_ticket": _support_ticket,
    "insurance_claim": _insurance_claim,
    "crm_record": _crm_record,
    "email": _email,
    "conversation": _conversation,
    "form": _form,
    "kg_triple": _kg_triple,
}


def build_seed(schema_name: str, rng: random.Random) -> tuple[str, dict]:
    """Return ``(clean_document_text, gold)`` for ``schema_name`` using ``rng``.

    Raises ``KeyError`` for an unregistered schema name (via
    :func:`schemaforge.registry.get_schema`) and for a registered schema that
    has no seed builder registered, so a missing schema fails loudly instead of
    being silently skipped.
    """
    get_schema(schema_name)  # raises KeyError for unknown schema names
    try:
        builder = _BUILDERS[schema_name]
    except KeyError:
        raise KeyError(f"no seed builder registered for schema {schema_name!r}") from None
    return builder(rng)
