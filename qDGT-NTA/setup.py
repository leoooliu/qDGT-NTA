import tkinter as tk
from tkinter import filedialog, messagebox
from predict import logD_prediction,logIE_neg_prediction, logIE_pos_prediction
from AD_check import logD_AD, logIE_neg_AD, logIE_pos_AD


# ===== 通用函数 =====
def run_prediction(pred_module, label):
    file_path = filedialog.askopenfilename(filetypes=[("CSV/Excel", "*.csv;*.xlsx")])
    if not file_path:
        return
    pred_output = filedialog.asksaveasfilename(defaultextension=".xlsx")
    if not pred_output:
        return
    pred_module.predict_smiles(file_path, pred_output)
    messagebox.showinfo("完成", f"{label} 预测完成，结果保存到 {pred_output}")


def run_ad(ad_module, label):
    file_path = filedialog.askopenfilename(filetypes=[("Excel 文件", "*.xlsx")])
    if not file_path:
        return
    ad_output = filedialog.asksaveasfilename(defaultextension=".xlsx")
    if not ad_output:
        return

    # 从模块中读取默认参数
    default_sim = getattr(ad_module, "DEFAULT_SIMILARITY_THRESHOLD", 0.35)
    default_count = getattr(ad_module, "DEFAULT_COUNT_THRESHOLD", 1)

    # 弹窗让用户选择是否修改阈值
    threshold_win = tk.Toplevel(root)
    threshold_win.title(f"{label} AD threshold")

    tk.Label(threshold_win, text="similarity_threshold:").grid(row=0, column=0, padx=5, pady=5)
    sim_entry = tk.Entry(threshold_win)
    sim_entry.insert(0, str(default_sim))
    sim_entry.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(threshold_win, text="count_threshold:").grid(row=1, column=0, padx=5, pady=5)
    count_entry = tk.Entry(threshold_win)
    count_entry.insert(0, str(default_count))
    count_entry.grid(row=1, column=1, padx=5, pady=5)

    def confirm():
        sim_val = float(sim_entry.get())
        count_val = int(count_entry.get())
        ad_module.evaluate_ad(file_path, ad_output,
                              similarity_threshold=sim_val,
                              count_threshold=count_val)
        messagebox.showinfo("Completed", f"{label} application domain check is completed and the results are saved to the {ad_output}")
        threshold_win.destroy()

    tk.Button(threshold_win, text="Run", command=confirm).grid(row=2, columnspan=2, pady=10)


# ===== 主界面 =====
root = tk.Tk()
root.title("qDGT-NTA.tool")

# logD 模型
tk.Label(root, text="model 1: -logD").pack(pady=5)
tk.Button(root, text="-logD predict", command=lambda: run_prediction(logD_prediction, "-logD")).pack(pady=2)
tk.Button(root, text="AD-D check", command=lambda: run_ad(logD_AD, "-logD")).pack(pady=2)

# logIE(+) 模型
tk.Label(root, text="model 2: logIE (+)").pack(pady=5)
tk.Button(root, text="logIE (+) predict", command=lambda: run_prediction(logIE_pos_prediction, "logIE (+)")).pack(pady=2)
tk.Button(root, text="AD-IE(+) check", command=lambda: run_ad(logIE_pos_AD, "logIE (+)")).pack(pady=2)

# logIE(-) 模型
tk.Label(root, text="model 3: logIE (-)").pack(pady=5)
tk.Button(root, text="logIE(-) predict", command=lambda: run_prediction(logIE_neg_prediction, "logIE (-)")).pack(pady=2)
tk.Button(root, text="AD-IE(-) check", command=lambda: run_ad(logIE_neg_AD, "logIE (-)")).pack(pady=2)

root.mainloop()
