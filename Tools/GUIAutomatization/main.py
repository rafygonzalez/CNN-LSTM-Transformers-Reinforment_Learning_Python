import os
import pyautogui

nomenclatures = {
    'KD': pyautogui.keyDown,
    'KU': pyautogui.keyUp,
    'TW': pyautogui.typewrite,
    'HK': pyautogui.hotkey,
    'SP': pyautogui.sleep,
    'CK': pyautogui.click,
    'MT': pyautogui.moveTo
}
dir = "actions/mac/apps"
labels = ["notes"]
batch_actions = []

# Iterate over the labels
for label in labels:
  # Get the list of file names in the train and test directories for the current label
  files = os.listdir(os.path.join(dir, label))

  # Iterate over the file names in the train and test directories
  for file in files:
    # Read the input and target sequences from the files
    with open(os.path.join(dir, label, file), "r") as f:
      input_sequence = f.readlines()
      action = []
      for text in input_sequence:
        text = text.strip('\n')
        action.append(text)
    
    batch_actions.append(action)

for action in batch_actions:
    for a in action:
        action_type = a[:2]
        action_value = a[3:].strip(']')
        if action_type in nomenclatures:
            if action_type == 'SP':
                nomenclatures[action_type](int(action_value))
            elif action_type == 'CK':
                nomenclatures[action_type](tuple(map(int, action_value.split(','))))
            else:
                nomenclatures[action_type](action_value)
        else:
            print(f'Invalid action type: {action_type}')
