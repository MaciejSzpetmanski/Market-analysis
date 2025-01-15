import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from tkcalendar import DateEntry

import pandas as pd
import random
import yfinance as yf

import data_pipeline as dp
import prediction_pipeline as pp
from datetime import datetime, timedelta


class StockPredictionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Stock Prediction App")
        self.geometry("1000x600")

        self.last_close_value = None
        self.df = None
        self.pred_date = None
        self.freshLoaded = False

        # -------- Główne ramki (layout) -----------
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        self.source_frame = ttk.Frame(main_frame)
        self.source_frame.pack(side="top", fill="x", padx=5, pady=5)

        self.details_frame = ttk.Frame(main_frame)
        self.details_frame.pack(side="top", fill="x", padx=5, pady=5)

        self.chart_frame = ttk.Frame(main_frame)
        self.chart_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.prediction_frame = ttk.Frame(main_frame, width=200)
        self.prediction_frame.pack(side="right", fill="y", padx=5, pady=5)

        # -------- Zmienne kontrolne -----------
        self.data_source = tk.StringVar(value="local")  # 'local' lub 'internet'

        # self.instruments_list = ["AAPL", "TSLA", "AMZN", "BTC-USD", "ETH-USD"]
        self.instruments_list = dp.load_object("names/names.pkl")
        self.instrument_var = tk.StringVar(value="AAPL")
        self.from_date_var = tk.StringVar(value="2023-01-01")
        self.to_date_var = tk.StringVar(value="2023-12-31")
        self.interval_var = tk.StringVar(value="1d")
        
        self.file_path_var = tk.StringVar()

        # -------- Elementy GUI -----------

        ttk.Label(self.source_frame, text="Wybierz źródło danych:").pack(side="left", padx=5)

        self.radio_local = ttk.Radiobutton(
            self.source_frame, text="Dane lokalne",
            variable=self.data_source, value="local",
            command=self.update_source_details
        )
        self.radio_local.pack(side="left", padx=5)

        self.radio_internet = ttk.Radiobutton(
            self.source_frame, text="Dane z internetu",
            variable=self.data_source, value="internet",
            command=self.update_source_details
        )
        self.radio_internet.pack(side="left", padx=5)

        self.local_frame = ttk.Frame(self.details_frame)
        self.internet_frame = ttk.Frame(self.details_frame)

        # Dane lokalne
        ttk.Label(self.local_frame, text="Ścieżka do pliku CSV:").pack(side="left", padx=5)
        self.file_entry = ttk.Entry(self.local_frame, textvariable=self.file_path_var, width=50)
        self.file_entry.pack(side="left", padx=5)
        self.browse_button = ttk.Button(self.local_frame, text="Wybierz plik", command=self.browse_file)
        self.browse_button.pack(side="left", padx=5)

        # Dane z internetu
        ttk.Label(self.internet_frame, text="Instrument:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.instrument_combo = ttk.Combobox(
            self.internet_frame,
            textvariable=self.instrument_var,
            values=self.instruments_list,
            width=12
        )
        self.instrument_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.internet_frame, text="Data od:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.from_date_entry = DateEntry(
            self.internet_frame,
            textvariable=self.from_date_var,
            date_pattern='yyyy-mm-dd',
            width=10
        )
        self.from_date_entry.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(self.internet_frame, text="Data do:").grid(row=0, column=4, padx=5, pady=5, sticky="e")
        self.to_date_entry = DateEntry(
            self.internet_frame,
            textvariable=self.to_date_var,
            date_pattern='yyyy-mm-dd',
            width=10
        )
        self.to_date_entry.grid(row=0, column=5, padx=5, pady=5)

        ttk.Label(self.internet_frame, text="Interwał:").grid(row=0, column=6, padx=5, pady=5, sticky="e")
        self.interval_combo = ttk.Combobox(
            self.internet_frame,
            textvariable=self.interval_var,
            values=["1d", "1h", "15m"],
            width=5
        )
        self.interval_combo.grid(row=0, column=7, padx=5, pady=5)

        # Przycisk "Wczytaj dane"
        self.load_data_button = ttk.Button(self.details_frame, text="Wczytaj dane", command=self.load_data)
        self.load_data_button.pack(side="left", padx=5)

        # Miejsce na wykres
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title("Wykres")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Panel boczny – predykcja
        ttk.Label(self.prediction_frame, text="Predykcja:", font=("Arial", 14, "bold")).pack(pady=5)
        self.prediction_label = tk.Label(self.prediction_frame, text="", font=("Arial", 12, "bold"))
        self.prediction_label.pack(pady=5)
        self.predict_button = ttk.Button(self.prediction_frame, text="Przewiduj", command=self.run_prediction)
        self.predict_button.pack(pady=5)

        self.update_source_details()

    def update_source_details(self):
        if self.data_source.get() == "local":
            self.internet_frame.pack_forget()
            self.local_frame.pack(side="left", fill="x", expand=True)
        else:
            self.local_frame.pack_forget()
            self.internet_frame.pack(side="left", fill="x", expand=True)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz plik z danymi",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def load_data(self):
        source = self.data_source.get()
        if source == "local":
            self.load_local_data()
        else:
            self.load_internet_data()
        print(self.df)
        self.freshLoaded = True

    def load_local_data(self):
        path = self.file_path_var.get()
        if not path:
            messagebox.showerror("Błąd", "Nie wybrano pliku CSV.")
            return

        # try:
        #     csv_data_converter.main()
        # except Exception as e:
        #     messagebox.showerror("Błąd", f"Konwersja CSV nie powiodła się:\n{e}")
        #     return

        # try:
        #     # TODO use prediction pipeline
        #     data_pipeline.main()
        # except Exception as e:
        #     messagebox.showerror("Błąd", f"Pipeline nie powiódł się:\n{e}")
        #     return

        try:
            self.df = pd.read_csv(path)
            self.update_chart(self.df)
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać finalnego CSV:\n{e}")

    def load_internet_data(self):
        instrument = self.instrument_var.get()
        from_date = self.from_date_var.get()
        to_date = self.to_date_var.get()
        interval = self.interval_var.get()

        if not instrument:
            messagebox.showerror("Błąd", "Nie wybrano instrumentu.")
            return

        try:
            self.df = yf.download(tickers=instrument, start=from_date, end=to_date, interval=interval)
            self.df.reset_index(inplace=True)
            self.df.columns = [c[0] for c in self.df.columns]
            self.df.columns = self.df.columns.str.lower()
            self.df["adjusted_close"] = self.df["close"]

            if 'datetime' in self.df.columns:
                self.df.rename(columns={'Datetime': 'date'}, inplace=True)
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się pobrać danych z yfinance:\n{e}")
            return

        self.update_chart(self.df)

    def update_chart(self, df):
        self.axes.clear()
        self.axes.set_title("Wykres (Close)")
        self.last_close_value = None

        if df is not None and not df.empty:
            if "date" not in df.columns:
                messagebox.showerror("Błąd", "Brak kolumny 'date' w danych.")
                return
            if "close" not in df.columns:
                messagebox.showerror("Błąd", "Brak kolumny 'close' w danych.")
                return

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=["date"], inplace=True)
            df.sort_values(by="date", inplace=True)
            self.pred_date = max(df['date']) + timedelta(days=1)
            print(self.pred_date)

            if df.empty:
                self.axes.text(0.5, 0.5, "Brak danych do wyświetlenia", ha="center", va="center")
            else:
                self.axes.plot(df["date"], df["close"], label='close', color='blue')
                self.last_close_value = df["close"].iloc[-1]
                self.axes.set_xlabel("data")
                self.axes.set_ylabel("cena [close]")
                self.axes.legend()
        else:
            self.axes.text(0.5, 0.5, "Brak danych", ha="center", va="center")

        self.canvas.draw()

    def run_prediction(self):
        name = self.instrument_var.get()
        if self.freshLoaded:
            try:
                x = pp.prepare_data_for_prediction(self.df, name)
                x = pp.merge_vector_with_pred_date(x, self.pred_date)
                predicted_value = pp.predict_value(x, name)[0]
                self.freshLoaded = False
            except Exception as e:
                messagebox.showerror("Błąd", str(e))
                return
        
        # predicted_value = self.mock_predict()
        if self.last_close_value is None:
            self.prediction_label.config(text="Brak ostatniej ceny do porównania", fg="black")
            return

        if predicted_value > self.last_close_value:
            self.prediction_label.config(
                text=f"Przewidywana cena: {predicted_value:.2f}",
                fg="green"
            )
        else:
            self.prediction_label.config(
                text=f"Przewidywana cena: {predicted_value:.2f}",
                fg="red"
            )

    # def mock_predict(self):
    #     # TODO attach model
    #     return random.uniform(100, 300)


def main():
    app = StockPredictionApp()
    app.mainloop()

if __name__ == "__main__":
    main()
