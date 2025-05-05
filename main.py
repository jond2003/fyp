from preprocess import preprocess_blogs
from classifier import train_classifier
from tests import validate_results
from textgen import generate_target_texts, generate_style_transfer

def prepare_data():
    preprocess_blogs()
    generate_target_texts()

def get_results():
    generate_style_transfer()
    validate_results()

def run_all():
    prepare_data()
    train_classifier()
    get_results()

# menu options
options = {
    "Prepare Dataset and Target Texts": prepare_data,
    "Train BERT Classifier": train_classifier,
    "Generate Style Transfer Results": get_results,
    "Evaluate Style Transfer": validate_results,
    "All": run_all
}

# menu loop for selecting tasks to perform
def main():
    while True:
        print("\n----- Personalised Style Transfer -----")
        for idx, option in enumerate(options.keys(), 1):
            print(f"{idx}. {option}")
        print("0. Exit")

        try:
            choice = int(input("\nSelect an option: "))
            if choice == 0:
                print("Exiting...")
                break
            elif 1 <= choice <= len(options):
                selected_option = list(options.values())[choice - 1]
                print(f"\nRunning: {list(options.keys())[choice - 1]}")
                selected_option()  # call the selected function
            else:
                print("Invalid selection. Please choose a valid option.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == '__main__':
    main()