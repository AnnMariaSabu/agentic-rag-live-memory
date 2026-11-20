"""Who may see which documents. (Reused from Project 1.)

The filter goes INTO the database query, so forbidden chunks are never
even fetched — safer than fetching everything and hiding some later.
"""

ROLES = ["Employee", "HR", "IT", "Admin"]


def search_filter(role):
    if role == "Employee":
        return {"access": "public"}
    if role == "HR":
        return {"$or": [{"access": "public"}, {"department": "hr"}]}
    if role == "IT":
        return {"$or": [{"access": "public"}, {"department": "it"}]}
    if role == "Admin":
        return None  # no filter: sees everything
    raise ValueError(f"Unknown role: {role}")
