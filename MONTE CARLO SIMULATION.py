import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from scipy.stats import skew, kurtosis
import io
from PIL import Image, ImageTk

def calculate_mean_daily_return(daily_returns):
    return np.mean(daily_returns)

def calculate_std_dev_daily_returns(daily_returns):
    return np.std(daily_returns)

def calculate_sharpe_ratio(mean_return, std_dev, risk_free_rate=0.01):
    return (mean_return - risk_free_rate) / std_dev

def calculate_var(prices, confidence_level=0.05):
    return np.percentile(prices, confidence_level * 100)

def calculate_cvar(prices, confidence_level=0.05):
    var = calculate_var(prices, confidence_level)
    return np.mean([price for price in prices if price <= var])

def calculate_max_drawdown(prices):
    cumulative_returns = np.maximum.accumulate(prices)
    drawdowns = (cumulative_returns - prices) / cumulative_returns
    return np.max(drawdowns)

def monte_carlo_simulation(initial_price, mean_return, std_dev, time_period, num_simulations):
    def simulate_once():
        prices = [initial_price]
        for _ in range(time_period):
            daily_return = np.random.normal(mean_return, std_dev)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        return prices

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda _: simulate_once(), range(num_simulations)))
    return results

class MonteCarloApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Monte Carlo Simulation for Crypto Asset Prices")
        self.geometry("800x600")
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Monte Carlo Simulation for Crypto Asset Prices", font=("Helvetica", 16)).pack(pady=10)

        self.initial_price_var = tk.DoubleVar()
        ttk.Label(self, text="Enter the initial price of the crypto asset:").pack(pady=5)
        ttk.Entry(self, textvariable=self.initial_price_var).pack(pady=5)

        self.time_period_var = tk.IntVar()
        ttk.Label(self, text="Enter the time period in days:").pack(pady=5)
        ttk.Entry(self, textvariable=self.time_period_var).pack(pady=5)

        self.num_simulations_var = tk.IntVar()
        ttk.Label(self, text="Enter the number of simulations:").pack(pady=5)
        ttk.Entry(self, textvariable=self.num_simulations_var).pack(pady=5)

        ttk.Button(self, text="Upload CSV", command=self.upload_csv).pack(pady=10)
        self.daily_returns_text = tk.Text(self, height=10)
        self.daily_returns_text.pack(pady=5)

        ttk.Button(self, text="Run Simulation", command=self.run_simulation).pack(pady=10)

    def upload_csv(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                data = pd.read_csv(file_path)
                daily_returns = data['Change %'][0:].str.replace('%', '').astype(float) / 100
                self.daily_returns = daily_returns.tolist()
                self.daily_returns_text.delete("1.0", tk.END)
                self.daily_returns_text.insert(tk.END, '\n'.join(map(str, self.daily_returns)))
                messagebox.showinfo("Success", "CSV file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file: {e}")

    def run_simulation(self):
        try:
            initial_price = self.initial_price_var.get()
            daily_returns = [float(x) for x in self.daily_returns_text.get("1.0", tk.END).strip().split("\n") if x]
            if not daily_returns:
                messagebox.showerror("Input Error", "Please enter or upload valid daily returns.")
                return

            mean_return = calculate_mean_daily_return(daily_returns)
            std_dev = calculate_std_dev_daily_returns(daily_returns)
            time_period = self.time_period_var.get()
            num_simulations = self.num_simulations_var.get()

            simulation_results = monte_carlo_simulation(initial_price, mean_return, std_dev, time_period, num_simulations)
            final_prices = [result[-1] for result in simulation_results]

            mean_final_price = np.mean(final_prices)
            std_final_price = np.std(final_prices)
            prob_loss = np.mean([price < initial_price for price in final_prices])
            percentile_5 = np.percentile(final_prices, 5)
            percentile_95 = np.percentile(final_prices, 95)
            skewness = skew(final_prices)
            kurtosis_value = kurtosis(final_prices)
            annualized_volatility = std_final_price * np.sqrt(252)
            sharpe_ratio = calculate_sharpe_ratio(mean_final_price, std_final_price)
            var = calculate_var(final_prices)
            cvar = calculate_cvar(final_prices)
            max_drawdown = calculate_max_drawdown(final_prices)
            prob_target_price = np.mean([price > initial_price * 1.5 for price in final_prices])

            self.display_statistics(mean_final_price, std_final_price, prob_loss, percentile_5, percentile_95, skewness, kurtosis_value,
                                    annualized_volatility, sharpe_ratio, var, cvar, max_drawdown, prob_target_price)
            self.display_histogram(final_prices, mean_final_price, percentile_5, percentile_95)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_statistics(self, mean_final_price, std_final_price, prob_loss, percentile_5, percentile_95, skewness, kurtosis_value,
                           annualized_volatility, sharpe_ratio, var, cvar, max_drawdown, prob_target_price):
        result_window = tk.Toplevel(self)
        result_window.title("Simulation Results")

        columns = ('Statistic', 'Value')
        tree = ttk.Treeview(result_window, columns=columns, show='headings')
        tree.heading('Statistic', text='Statistic')
        tree.heading('Value', text='Value')

        data = [
            ('Mean final price', f"${mean_final_price:.2f}"),
            ('Standard deviation of final prices', f"${std_final_price:.2f}"),
            ('Probability of a loss', f"{prob_loss * 100:.2f}%"),
            ('5th percentile of final prices', f"${percentile_5:.2f}"),
            ('95th percentile of final prices', f"${percentile_95:.2f}"),
            ('Skewness of final prices', f"{skewness:.2f}"),
            ('Kurtosis of final prices', f"{kurtosis_value:.2f}"),
            ('Annualized volatility', f"{annualized_volatility:.2f}"),
            ('Sharpe ratio', f"{sharpe_ratio:.2f}"),
            ('Value at Risk (VaR)', f"${var:.2f}"),
            ('Conditional VaR (CVaR)', f"${cvar:.2f}"),
            ('Maximum drawdown', f"{max_drawdown:.2f}"),
            ('Probability of reaching 150% initial price', f"{prob_target_price * 100:.2f}%"),
        ]

        for item in data:
            tree.insert('', tk.END, values=item)

        tree.pack(pady=10, padx=10, fill='both', expand=True)

    def display_histogram(self, final_prices, mean_final_price, percentile_5, percentile_95):
        fig = plt.figure(figsize=(10, 6))
        plt.hist(final_prices, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Monte Carlo Simulation of Crypto Asset Prices')
        plt.xlabel('Final Price')
        plt.ylabel('Frequency')
        plt.axvline(mean_final_price, color='r', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_final_price:.2f}')
        plt.axvline(percentile_5, color='g', linestyle='dashed', linewidth=2, label=f'5th Percentile: ${percentile_5:.2f}')
        plt.axvline(percentile_95, color='y', linestyle='dashed', linewidth=2, label=f'95th Percentile: ${percentile_95:.2f}')
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img = Image.open(img)
        img = ImageTk.PhotoImage(img)

        hist_window = tk.Toplevel(self)
        hist_window.title("Simulation Histogram")

        panel = ttk.Label(hist_window, image=img)
        panel.image = img
        panel.pack(pady=10)

if __name__ == "__main__":
    app = MonteCarloApp()
    app.mainloop()