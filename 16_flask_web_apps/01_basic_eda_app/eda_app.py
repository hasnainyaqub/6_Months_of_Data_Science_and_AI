from flask import Flask, request, render_template_string
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
PLOT_FOLDER = 'static'
os.makedirs(PLOT_FOLDER, exist_ok=True)

def load_and_process_dataset(name):
    df = sns.load_dataset(name)

    if name == 'titanic':
        if 'deck' in df.columns:
            df = df.drop(columns=['deck'])
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def create_plots(df):
    pairplot_path = os.path.join(PLOT_FOLDER, 'pairplot.png')
    hist_path = os.path.join(PLOT_FOLDER, 'hist.png')

    # Pairplot
    try:
        sns.pairplot(df.select_dtypes(include='number'))
        plt.savefig(pairplot_path)
        plt.clf()
    except:
        with open(pairplot_path, 'wb') as f:
            pass

    # Histogram
    df.select_dtypes(include='number').hist(figsize=(10, 6))
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.clf()

    return pairplot_path, hist_path

@app.route('/', methods=['GET', 'POST'])
def index():
    dataset_options = ['iris', 'titanic', 'tips', 'diamonds']
    selected_dataset = request.form.get('dataset')
    data_loaded = False

    if request.method == 'POST' and selected_dataset:
        data_loaded = True
        df = load_and_process_dataset(selected_dataset)
        pairplot_path, hist_path = create_plots(df)

        full_data_html = df.to_html(classes='data', header="true", index=False)
        desc = df.describe().to_html(classes='data', header="true")
        shape = df.shape
    else:
        full_data_html = desc = shape = pairplot_path = hist_path = None

    return render_template_string("""
    <html>
    <head>
        <title>Seaborn EDA App</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
                min-height: 100vh;
                background-color: #f0f0f0;
            }
            .container {
                width: 95%;
                max-width: 1200px;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            select, button {
                padding: 10px 15px;
                font-size: 16px;
                margin: 10px;
            }
            table.data {
                margin: 0 auto 20px;
                border-collapse: collapse;
                width: 100%;
            }
            table.data th, table.data td {
                border: 1px solid #ccc;
                padding: 8px;
            }
            .scrollable-table {
                max-height: 300px;
                overflow-y: scroll;
                border: 1px solid #ccc;
                margin-bottom: 20px;
            }
            img {
                max-width: 100%;
                height: auto;
                margin-bottom: 20px;
                border-radius: 6px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Seaborn Dataset Explorer</h1>

            <form method="POST">
                <label for="dataset">Select a dataset:</label>
                <select name="dataset" id="dataset">
                    {% for option in dataset_options %}
                        <option value="{{ option }}" {% if option == selected_dataset %}selected{% endif %}>{{ option }}</option>
                    {% endfor %}
                </select>
                <button type="submit">Load</button>
            </form>

            {% if data_loaded %}
                <h2>Dataset: {{ selected_dataset }} (Shape: {{ shape }})</h2>

                <h2>Full Dataset</h2>
                <div class="scrollable-table">
                    {{ full_data_html|safe }}
                </div>

                <h2>Describe</h2>
                {{ desc|safe }}

                <h2>Pairplot</h2>
                <img src="{{ url_for('static', filename='pairplot.png') }}">

                <h2>Histogram</h2>
                <img src="{{ url_for('static', filename='hist.png') }}">
            {% endif %}
        </div>
    </body>
    </html>
    """, dataset_options=dataset_options, selected_dataset=selected_dataset,
       data_loaded=data_loaded, full_data_html=full_data_html,
       desc=desc, shape=shape)

if __name__ == '__main__':
    app.run(debug=True)
