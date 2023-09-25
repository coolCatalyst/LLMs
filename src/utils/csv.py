import csv
def save_to_csv(messages, filename):
    """
    Save a list of messages to a csv file.
    """
    with open(filename, "w", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["ID", "Message"])
        csv_writer.writerows(messages)
