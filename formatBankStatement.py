import re
import csv
import sys

def clean_amount(s):
    if not s:
        return ''
    return s.replace(',', '')

def main():
    # Citește conținutul fișierului extras.txt
    try:
        with open('extras.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("Eroare: Fișierul 'extras.txt' nu a fost găsit în directorul curent.", file=sys.stderr)
        sys.exit(1)

    lines = text.splitlines()
    transactions = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = re.match(r'^(\d{2}\.\d{2}\.\d{4}) (\d{2}\.\d{2}\.\d{4}) (.*)', line)
        if match:
            data_inreg = match.group(1)
            data_tranz = match.group(2)
            descriere = match.group(3).strip()
            j = i + 1
            amount = None
            while j < len(lines):
                next_line = lines[j].strip()
                if re.match(r'^\d{2}\.\d{2}\.\d{4} \d{2}\.\d{2}\.\d{4}', next_line):
                    break
                if re.match(r'^[\d,]+\.?\d*$', next_line):
                    amount = clean_amount(next_line)
                    j += 1
                    break
                if next_line:
                    descriere += " " + next_line
                j += 1
            is_credit = 'SALARIU' in descriere or ('THALES ROMANIA' in descriere and 'OPH' in descriere)
            if is_credit:
                debit = ''
                credit = amount if amount else ''
            else:
                debit = amount if amount else ''
                credit = ''
            transactions.append([data_inreg, data_tranz, descriere, debit, credit])
            i = j
        else:
            i += 1

    # Scrie în fișierul out.csv
    with open('out.csv', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Data înregistrare', 'Data tranzacției', 'Descriere', 'Sumă debit (RON)', 'Sumă credit (RON)'])
        writer.writerows(transactions)

    print(f"✅ CSV generat cu succes în fișierul 'out.csv' ({len(transactions)} tranzacții).")

if __name__ == '__main__':
    main()