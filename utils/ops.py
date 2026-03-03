def normalize_ec(ec_str: str) -> str:
    ec_items = ec_str.split(";")
    ec_level3 = set()

    for ec in ec_items:
        ec = ec.strip()
        if not ec.startswith("EC:"):
            continue

        parts = ec.replace("EC:", "").split(".")
        if len(parts) >= 3:
            ec_level3.add("EC:" + ".".join(parts[:3])+".*")

    return ";".join(sorted(ec_level3))
