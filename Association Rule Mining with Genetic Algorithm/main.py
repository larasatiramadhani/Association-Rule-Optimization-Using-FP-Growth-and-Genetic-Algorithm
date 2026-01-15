# =====================================================
# IMPORT
# =====================================================
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import ast
import argparse
import random
from itertools import combinations
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ====================================================
# SETIING PARAMETER
# =====================================================
minsup = 0.01 #minimum support
max_k = 4 #maximum frequent itemset & rule
population_size = 100 #maximum population size
MIN_CONF=0.2
MIN_LIFT=1.0
TV=0.5
CR=0.9 
MR=0.9
 
# =====================================================
# FP-GROWTH PHASE (ASLI – TIDAK DIUBAH)
# =====================================================
def fpgrowth_phase(transactions, minsup, max_k):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    print('Data setelah di encode:', df_encoded.head())
    print('-' * 100)

    freq_itemsets = fpgrowth(df_encoded, min_support=minsup, use_colnames=True)
    print('Data setelah dijalankan fp-growth:', freq_itemsets)
    print('-' * 100)

    n = len(transactions)
    F = {}

    for _, row in freq_itemsets.iterrows():
        itemset = frozenset(row['itemsets'])
        k = len(itemset)
        if k > max_k or k < 1:
            continue

        count = int(round(row['support'] * n))
        if k not in F:
            F[k] = {}
        F[k][itemset] = count

    for k in sorted(F.keys()):
        print(f"F{k} ditemukan {len(F[k])} frequent itemsets")

    return F, int(minsup * n)


# =====================================================
# EVALUATION SUPPORT, CONFIDENCE & LIFT
# =====================================================
def evaluate_confidence_support_lift(rule, transactions):
    antecedent = rule["antecedent"]
    consequent = rule["consequent"]
    n = len(transactions)

    if n == 0:
        rule["support"] = 0
        rule["confidence"] = 0
        rule["lift"] = 0
        return rule

    antecedent_count = sum(1 for t in transactions if antecedent.issubset(t))
    consequent_count = sum(1 for t in transactions if consequent.issubset(t))
    both_count = sum(1 for t in transactions if (antecedent | consequent).issubset(t))

    support = both_count / n
    confidence = (both_count / antecedent_count) if antecedent_count > 0 else 0
    support_consequent = consequent_count / n
    lift = (confidence / support_consequent) if support_consequent > 0 else 0

    rule["support"] = round(support, 6)
    rule["confidence"] = round(confidence, 6)
    rule["lift"] = round(lift, 6)

    return rule

# ==================================
# NOVELTY MEASURE AS FITNESS FUNCTION
# ==================================
def novelty_measure(rule1, rule2):
    g1 = set(rule1["antecedent"]) | set(rule1["consequent"])
    g2 = set(rule2["antecedent"]) | set(rule2["consequent"])
    k = len(g1 & g2)
    denom = len(g1) + len(g2)

    if denom == 0:
        return 0.0
    return max(0.0, min(1.0, (denom - 2 * k) / denom))


# =====================================================
# PEMBENTUKAN POPULASI AWAL
# =====================================================
def initialize_population(F, transactions, max_k, population_size):
    all_possible_rules = []

    for q in range(2, max_k + 1):
        if q not in F:
            continue

        for Fi in F[q].keys():
            items = list(Fi)
            for r in range(1, len(items)):
                for subset in combinations(items, r):
                    A = set(subset)
                    B = Fi - A

                    rule = {
                        "antecedent": A,
                        "consequent": B,
                        "level": q,
                        "novelty": 0.0
                    }
                    rule = evaluate_confidence_support_lift(rule, transactions)
                    all_possible_rules.append(rule)

    if len(all_possible_rules) > population_size:
        population = random.sample(all_possible_rules, population_size)
    else:
        population = all_possible_rules

    return population

# =====================================================
# CROSSOVER
# =====================================================
def one_point_crossover(p1, p2, CR):
    c1, c2 = p1.copy(), p2.copy()

    if random.random() < CR:
        side_to_crossover = random.choice([0, 1])

        if side_to_crossover == 0:
            a1, a2 = list(p1["antecedent"]), list(p2["antecedent"])
            len1, len2 = len(a1), len(a2)

            if len1 == 0 or len2 == 0:
                pass

            elif len1 == 1 and len2 == 1:
                c1["antecedent"] = set(a2)
                c2["antecedent"] = set(a1)

            elif min(len1, len2) > 1:
                cut_a = random.randint(1, min(len1, len2) - 1)
                c1["antecedent"] = set(a1[:cut_a] + a2[cut_a:])
                c2["antecedent"] = set(a2[:cut_a] + a1[cut_a:])

            else:
                if len1 == 1:
                    c1["antecedent"] = {a2[0]}
                    c2["antecedent"] = {a1[0]} | set(a2[1:])
                else:
                    c1["antecedent"] = {a2[0]} | set(a1[1:])
                    c2["antecedent"] = {a1[0]}

        else:
            b1, b2 = list(p1["consequent"]), list(p2["consequent"])
            len1, len2 = len(b1), len(b2)

            if len1 == 0 or len2 == 0:
                pass

            elif len1 == 1 and len2 == 1:
                c1["consequent"] = set(b2)
                c2["consequent"] = set(b1)

            elif min(len1, len2) > 1:
                cut_b = random.randint(1, min(len1, len2) - 1)
                c1["consequent"] = set(b1[:cut_b] + b2[cut_b:])
                c2["consequent"] = set(b2[:cut_b] + b1[cut_b:])

            else:
                if len1 == 1:
                    c1["consequent"] = {b2[0]}
                    c2["consequent"] = {b1[0]} | set(b2[1:])
                else:
                    c1["antecedent"] = {b2[0]} | set(b1[1:])
                    c2["antecedent"] = {b1[0]}

    return c1, c2


# =====================================================
# MUTATION
# =====================================================
def mutation(chrom, all_items, MR):
    if random.random() < MR:
        side_to_mutation = random.choice([0, 1])
        target_key = "antecedent" if side_to_mutation == 0 else "consequent"
        target_set = chrom[target_key].copy()
        target_set = set(target_set)

        if len(all_items) == 0:
            return chrom

        if not target_set:
            mutation_type = "ADD"
        else:
            mutation_type = random.choice(["ADD", "REPLACE"])

        if mutation_type == "ADD":
            available_items = list(set(all_items) - target_set)
            if available_items:
                target_set.add(random.choice(available_items))

        elif mutation_type == "REPLACE":
            item_to_replace = random.choice(list(target_set))
            available_items = list(set(all_items) - target_set)
            if available_items:
                target_set.remove(item_to_replace)
                target_set.add(random.choice(available_items))

        chrom[target_key] = target_set

    return chrom


# =====================================================
# GA MAIN
# =====================================================
def GA_Discovery_with_lift(F, transactions, max_k, MIN_CONF, MIN_LIFT, TV, CR, MR):
    w = sum(len(F[k]) for k in F)
    MC = int((2**max_k - 2) * w)
    population = initialize_population(F, transactions, max_k, population_size)
    CHRiPDNAR = []

    all_items = list({item for level in F.values() for it in level for item in it})
    for _ in range(MC):
        new_pop = []

        for _ in range(0, max(1, len(population)//2)):
            k = min(3, len(population))
            p1 = max(random.sample(population, k), key=lambda x: x["lift"])
            p2 = max(random.sample(population, k), key=lambda x: x["lift"])

            c1, c2 = one_point_crossover(p1, p2, CR)
            c1 = mutation(c1, all_items, MR)
            c2 = mutation(c2, all_items, MR)

            for child in [c1, c2]:
                child = evaluate_confidence_support_lift(child, transactions)
                if child["antecedent"] == child["consequent"]:
                    continue
                if child["antecedent"].intersection(child["consequent"]):
                    continue

                if child["confidence"] >= MIN_CONF and child['lift']>MIN_LIFT:
                    # hitung novelty
                    if CHRiPDNAR:
                        psi_vals = [novelty_measure(child, old) for old in CHRiPDNAR]
                        child["novelty"] = min(psi_vals)
                    else:
                        child["novelty"] = 1.0

                    if child["novelty"] >= TV:
                        CHRiPDNAR.append(child)
                        new_pop.append(child)

        # elitist reproduction
        if population:
            best = max(population, key=lambda x: x["novelty"])
            new_pop.append(best)

        if not new_pop:
            new_pop = population.copy()

        population = new_pop

    if not CHRiPDNAR:
        print("Tidak ada aturan novel yang ditemukan.")
        return pd.DataFrame()

    return pd.DataFrame([
        {
            "antecedent": ', '.join(sorted(r["antecedent"])),
            "consequent": ', '.join(sorted(r["consequent"])),
            "confidence": r["confidence"],
            "lift": r["lift"],
            "novelty": round(r["novelty"], 3)
        }
        for r in CHRiPDNAR
    ]).sort_values(by="lift", ascending=False).reset_index(drop=True)


# =====================================================
# MAIN
# =====================================================
def main(path_transaksi, path_menu):
    df_transaksi_detail = pd.read_excel(path_transaksi)
    df_menu = pd.read_excel(path_menu)

    df_menu['kode'] = df_menu['kode'].astype(str).str.strip().str.upper()
    df_menu['nama'] = df_menu['nama'].astype(str).str.strip().str.lower()
    df_menu['Deskripsi'] = df_menu['Deskripsi'].astype(str).str.strip().str.lower()

    df_transaksi_detail = df_transaksi_detail[['nonota', 'kodebrg', 'jumlah', 'harga']]
    df_transaksi_detail['kodebrg'] = df_transaksi_detail['kodebrg'].astype(str).str.strip().str.upper()

    transaksi_merged = df_transaksi_detail.merge(df_menu, left_on="kodebrg", right_on="kode")
    transactions = transaksi_merged.groupby('nonota')['nama'].apply(list).tolist()

    F, _ = fpgrowth_phase(transactions,minsup, max_k)
    df_result = GA_Discovery_with_lift(F, transactions, max_k, MIN_CONF, MIN_LIFT, TV, CR, MR)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df_result)

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaksi", default="Data/dtl_jual.xlsx")
    parser.add_argument("--menu", default="Data/deskripsi menu.xlsx")

    args = parser.parse_args()
    main(args.transaksi, args.menu)
