import asyncio
import json
import joblib
import websockets
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

async def monitor_stream(url,model):
    # Load the trained model from train.py
    try:
      model = joblib.load(model)
    except FileNotFoundError:
      logging.error(f"Model file {model} not found")
      exit(1)
    async with websockets.connect(url) as websocket:
        
        event_counter = 1
        current_event = []
        last_seqid = None
        seqid_delta = 75  # Define the amount messages should be from each other to be grouped as on conversation

        try:
            while True:
                message = await websocket.recv()
                message_data = json.loads(message)

                # Extract the message content and seqid, since that's all we use for logic
                message_text = message_data['message']
                current_seqid = message_data['seqid']
                features = [message_text]
                prediction = model.predict(features)

                if prediction[0] == 1:
                    logging.info("Detected a meeting-related message")
                    if last_seqid is None or abs(current_seqid - last_seqid) <= seqid_delta:
                        if last_seqid is not None: # this is not the most elegant logic I've ever written
                            logging.info(f"Sequence delta is {abs(current_seqid - last_seqid)}")
                        current_event.append(message_data)
                        last_seqid = current_seqid
                    else:
                        save_event(current_event, event_counter)
                        logging.info("Saving event")
                        event_counter += 1
                        current_event = [message_data]
                        last_seqid = current_seqid
                else:
                    # logging.info("Non-meeting message detected")
                    if current_event and abs(current_seqid - last_seqid) > seqid_delta:
                        save_event(current_event, event_counter)
                        logging.info("Saving event file due to hitting max grouping amount")
                        event_counter += 1
                        current_event = []
                        last_seqid = None
        except websockets.ConnectionClosed:
            logging.info("WebSocket connection closed")
        finally:
            if current_event:
                save_event(current_event, event_counter)
                logging.info("Saving the last event before exiting")

def save_event(event, event_counter):
    if not os.path.exists('results'):
        os.makedirs('results')
    
    event_file = f'results/event_{event_counter:04d}.json' # write the files to zero leading 4 digit number.json
    with open(event_file, 'w') as f:
        json.dump({"lines": event}, f, indent=4)
    logging.info(f"Event saved to {event_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor a WebSocket stream for messages about meeting requests.")
    parser.add_argument("url", type=str, help="The WebSocket URL to connect to (example: ws://127.0.0.1:8000)")
    parser.add_argument("model", type=str, help="The trained model file to load (example: meeting_request_classifier.pkl)")

    args = parser.parse_args()

    asyncio.run(monitor_stream(args.url, args.model))