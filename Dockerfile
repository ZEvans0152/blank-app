FROM python:3.12-slim

WORKDIR /app
COPY stock_alert_bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stock_alert_bot/ stock_alert_bot/

# Keep subscriptions/watchlists/cooldowns on a volume so they survive updates.
ENV STATE_FILE=/data/state.json
VOLUME /data

CMD ["python", "-m", "stock_alert_bot"]
