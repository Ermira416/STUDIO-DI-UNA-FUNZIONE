import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os
import random

x = sp.symbols('x')

a = random.choice([n for n in range(-5, 6) if n != 0])
b = random.randint(-5, 5)
c = random.randint(-5, 5)

modelli_funzione = [
    a*x**2 + b*x + c,          
    a*x**3 + b*x**2 + c,       
    (a*x + b) / (x**2 + 4)     
]

y = random.choice(modelli_funzione)

# --- Cartella di destinazione ---
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
cartella = os.path.join(desktop, "studio_funzione")
os.makedirs(cartella, exist_ok=True)

# --- Analisi della Funzione ---
dominio = sp.calculus.util.continuous_domain(y, x, sp.Reals)
lim_sx = sp.limit(y, x, -sp.oo)
lim_dx = sp.limit(y, x, sp.oo)
derivata = sp.diff(y, x)

# --- CORREZIONE DISEQUAZIONI ---
# solve_univariate_inequality gestisce correttamente i segni delle funzioni fratte
crescenza = sp.solve_univariate_inequality(derivata > 0, x, relational=False)
decrescenza = sp.solve_univariate_inequality(derivata < 0, x, relational=False)

# --- Salvataggio Dati su File ---
with open(os.path.join(cartella, "studio_funzione.txt"), "w", encoding="utf-8") as file:
    file.write(f"FUNZIONE ESTRATTA: {y}\n")
    file.write("----------------------------------------\n")
    file.write(f"Dominio: {dominio}\n")
    file.write(f"Limite per x → -∞: {lim_sx}\n")
    file.write(f"Limite per x → +∞: {lim_dx}\n")
    file.write(f"Derivata prima: {derivata}\n")
    file.write(f"Intervalli di crescenza: {crescenza}\n")
    file.write(f"Intervalli di decrescenza: {decrescenza}\n")

# --- Generazione Grafico Intelligente (Autoadattivo) ---
f = sp.lambdify(x, y, "numpy")

# Trova i punti stazionari (dove f'(x) = 0) per capire dove centrare il grafico
punti_critici = sp.solve(derivata, x)
punti_reali = []
for p in punti_critici:
    try:
        val_float = float(p.evalf())
        # Controlla se la soluzione è effettivamente reale (non complessa)
        if p.is_real or np.isreal(val_float):
            punti_reali.append(val_float)
    except (TypeError, ValueError):
        continue

# Se ci sono punti critici, centra il grafico lì, altrimenti usa lo 0
centro_x = punti_reali[0] if punti_reali else 0

# Imposta un raggio di visualizzazione dinamico attorno al centro della funzione
raggio_x = 5 
x_vals = np.linspace(centro_x - raggio_x, centro_x + raggio_x, 500)
y_vals = f(x_vals)

plt.figure(figsize=(9, 6))
plt.plot(x_vals, y_vals, label=f"$y = {sp.latex(y)}$", color="crimson", linewidth=2)

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title(f"Studio Grafico Dinamico", fontsize=14)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

# --- CORREZIONE MARGINE ASSE Y ---
y_min, y_max = np.min(y_vals), np.max(y_vals)
margine = (y_max - y_min) * 0.1 if y_max != y_min else 1
plt.ylim(y_min - margine, y_max + margine)  # <--- Cambiato il secondo '-' in '+'

plt.savefig(os.path.join(cartella, "grafico.png"), dpi=300)
plt.close()

print(f"Analisi completata con successo per: {y}")