from flask import Flask, render_template_string

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Bybit Bot Log</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {
            background: #111;
            color: #0f0;
            font-family: monospace;
            padding: 20px;
        }
        pre {
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <h2>Bybit Auto Trading Log</h2>
    <pre>{{ log }}</pre>
</body>
</html>
"""

@app.route("/")
def home():
    try:
        with open("trade.log", "r", encoding="utf-8") as f:
            log = f.read()[-20000:]  # 마지막 부분만 표시
    except:
        log = "No log found"

    return render_template_string(HTML, log=log)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)