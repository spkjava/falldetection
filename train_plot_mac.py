import os
import cv2
import pickle
import re
import pandas as pd
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix

# --- ค่าตั้งต้น ---
ROOT_DIR = "fall_detection"
OUT_CSV = "skeleton_dataset_from_sequences.csv"
MODEL = "fall_model_from_sequences.pkl"

# --- นิยามชื่อฟีเจอร์ 132 ตัว (33 จุด × x,y,z,v) ---
POSE_COLS = [f"{axis}{i}" for i in range(33) for axis in ["x", "y", "z", "v"]]

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_sequence_id(filename):
    """ดึง sequence_id จากชื่อไฟล์ (เหมือนเดิม)"""
    match = re.match(r'([a-zA-Z0-9]+-\d+-[a-zA-Z0-9]+-[a-zA-Z0-9]+)', filename)
    if match:
        return match.group(1)
    return filename

def process_image_sequences(split_dir):
    ann_path = os.path.join(split_dir, "_annotations.csv")
    if not os.path.exists(ann_path):
        print(f"[INFO] ข้าม {split_dir} (ไม่พบ _annotations.csv)")
        return []

    df = pd.read_csv(ann_path)
    required_cols = {"filename", "class"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{ann_path} ต้องมีคอลัมน์ {required_cols}")

    df['sequence_id'] = df['filename'].apply(get_sequence_id)
    grouped = df.groupby('sequence_id')
    print(f"[{os.path.basename(split_dir)}] พบ {len(grouped)} ซีเควนซ์ (วิดีโอ)")

    all_features = []
    miss, nopose = 0, 0
    img_dir = os.path.join(split_dir, "images")

    for seq_id, group_df in grouped:
        sorted_group = group_df.sort_values(by='filename').reset_index()
        for i, row in sorted_group.iterrows():
            img_path = os.path.join(img_dir, row['filename'])
            img = cv2.imread(img_path)
            if img is None:
                miss += 1
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_img)

            if not results.pose_landmarks:
                nopose += 1
                continue

            feats = []
            for lm in results.pose_landmarks.landmark:
                feats.extend([lm.x, lm.y, lm.z, lm.visibility])

            # ตรวจสอบว่าได้ 132 features จริง
            if len(feats) != 132:
                print(f"[WARN] ข้าม {img_path} (ได้ {len(feats)} features)")
                continue

            label = "fall" if "fall" in str(row["class"]).lower() else "person"
            feats.append(label)
            all_features.append(feats)

    print(
        f"[{os.path.basename(split_dir)}] ประมวลผลเสร็จสิ้น: "
        f"พบข้อมูล={len(all_features)} | รูปหาย={miss} | ไม่เจอท่าทาง={nopose}"
    )
    return all_features

# ---------------------------------------------------------------------
#  B) ใช้ split ตามโฟลเดอร์ train / valid / test (ไม่ใช้ train_test_split)
# ---------------------------------------------------------------------
def build_dataset():
    """
    สร้าง DataFrame แยกตาม split:
      - train_df   จากโฟลเดอร์ ROOT_DIR/train
      - valid_df   จาก ROOT_DIR/valid (ถ้ามี)
      - test_df    จาก ROOT_DIR/test  (ถ้ามี)

    และสร้างไฟล์ OUT_CSV ที่รวมทุก split (สำหรับ debug/วิเคราะห์ข้อมูล)
    """
    dfs = {}
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(ROOT_DIR, split)
        if os.path.isdir(split_path):
            rows = process_image_sequences(split_path)
            if rows:
                cols = POSE_COLS + ["label"]
                df_split = pd.DataFrame(rows, columns=cols)
                dfs[split] = df_split

                # บันทึก CSV แยกแต่ละ split (ไว้ใช้ดู/ตรวจสอบ)
                split_csv = f"skeleton_dataset_{split}.csv"
                df_split.to_csv(split_csv, index=False)
                print(
                    f"✅ บันทึก {split} dataset {len(df_split)} แถว -> {split_csv}\n"
                    f"   จำนวนข้อมูลแต่ละคลาส (split={split}):\n{df_split['label'].value_counts()}\n"
                )

    if not dfs:
        print("❌ ไม่พบข้อมูลสำหรับสร้าง Dataset")
        return None

    # รวมทุก split ไว้เป็นไฟล์รวม (optional)
    df_all = pd.concat(dfs.values(), ignore_index=True)
    df_all.to_csv(OUT_CSV, index=False)
    print(
        f"✅ บันทึก dataset รวม {len(df_all)} แถว -> {OUT_CSV}\n"
        f"   จำนวนข้อมูลแต่ละคลาส (รวมทุก split):\n{df_all['label'].value_counts()}\n"
    )

    return dfs

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, tag):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix ({tag})')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    save_path = f'{tag}_confusion_matrix.png'
    plt.savefig(save_path)
    print(f"✅ บันทึกกราฟ Confusion Matrix -> {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, tag):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    sorted_df = importance_df.sort_values(by='importance', ascending=False)
    top_n_df = sorted_df.nlargest(25, 'importance')

    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=top_n_df, palette='viridis')
    plt.title(f'Top 25 Feature Importance ({tag})')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature (Landmark Parameter)')
    plt.tight_layout()
    save_path = f'{tag}_feature_importance.png'
    plt.savefig(save_path)
    print(f"✅ บันทึกกราฟ Feature Importance -> {save_path}")
    plt.close()

def plot_rf_learning_curves(estimator, X, y, tag):
    print("\n[INFO] กำลังสร้าง Learning Curves (อาจใช้เวลาสักครู่)...")
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='f1_weighted'
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
        plt.title(f'Learning Curves ({tag})')
        plt.xlabel('Training examples')
        plt.ylabel('Score (F1 Weighted)')
        plt.legend(loc='best')
        plt.grid(True)
        save_path = f'{tag}_learning_curves.png'
        plt.savefig(save_path)
        print(f"✅ บันทึกกราฟ Learning Curves -> {save_path}")
        plt.close()
    except Exception as e:
        print(f"[WARN] ไม่สามารถสร้าง Learning Curves: {e}")

def train(dfs):
    """
    เวอร์ชัน B:
      - ใช้ df จากโฟลเดอร์ train เป็นชุดฝึก
      - ใช้ df จาก valid + test รวมกันเป็นชุดทดสอบ (evaluation)
      - ไม่ใช้ train_test_split 70:30 อีกต่อไป
    """
    if "train" not in dfs:
        print("❌ ไม่พบ split 'train' ใน dataset")
        return

    df_train = dfs["train"]
    df_valid = dfs.get("valid")
    df_test  = dfs.get("test")

    # --- เตรียม train set ---
    X_train = df_train[POSE_COLS].values
    y_train = df_train["label"].values

    # --- เตรียม eval set: valid + test ---
    eval_dfs = []
    if df_valid is not None:
        eval_dfs.append(df_valid)
    if df_test is not None:
        eval_dfs.append(df_test)

    if not eval_dfs:
        print("❌ ไม่พบทั้ง valid และ test สำหรับใช้ประเมินโมเดล")
        return

    df_eval = pd.concat(eval_dfs, ignore_index=True)
    X_eval = df_eval[POSE_COLS].values
    y_eval = df_eval["label"].values

    print(
        f"[INFO] ใช้ train จาก split 'train' จำนวน {len(df_train)} แถว\n"
        f"      ใช้ eval จาก split 'valid' + 'test' จำนวน {len(df_eval)} แถว"
    )
    print("จำนวนข้อมูลแต่ละคลาสใน eval set:\n", df_eval["label"].value_counts())

    feature_names = POSE_COLS
    tag = MODEL.split('.')[0]

    # --- สร้าง estimator (ยังไม่ฝึก) ---
    clf_estimator = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        criterion="gini",
        class_weight={"fall": 3.0, "person": 1.0},
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    # --- Learning Curves ใช้ train set ---
    plot_rf_learning_curves(clf_estimator, X_train, y_train, tag)

    # --- ฝึกโมเดล ---
    print("\n[INFO] กำลังฝึกโมเดล (ใช้เฉพาะ split 'train') ...")
    clf_estimator.fit(X_train, y_train)

    # --- ประเมินบน eval set (valid + test) ---
    y_pred = clf_estimator.predict(X_eval)

    print("\n=== Classification Report (eval: valid+test) ===")
    print(classification_report(y_eval, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_eval, y_pred))

    # --- กราฟประเมินผล ---
    print("\n[INFO] กำลังสร้างกราฟประเมินผล...")
    plot_confusion_matrix_heatmap(y_eval, y_pred, clf_estimator.classes_, tag)
    plot_feature_importance(clf_estimator, feature_names, tag)

    # --- บันทึกโมเดล ---
    with open(MODEL, "wb") as f:
        pickle.dump(clf_estimator, f)
    print(f"✅ บันทึกโมเดล -> {MODEL}")

if __name__ == "__main__":
    dfs = build_dataset()
    if dfs is not None:
        train(dfs)
