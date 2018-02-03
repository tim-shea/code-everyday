import re
import yaml


class Character:
    def __init__(self, name):
        self.name = name


class Event:
    def __init__(self, template_text):
        self.template_text = template_text

    def format(self, context):
        tag_pattern = re.compile('<[a-z]+>')
        formatted_text = self.template_text
        for tag in re.finditer(tag_pattern, formatted_text):
            print(tag)


if __name__ == "__main__":
    result = yaml.load(open('events.yml', 'r'))
    event = Event(result['events'][0]['template_text'])
    event.format(None)
