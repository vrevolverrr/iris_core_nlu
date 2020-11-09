from datetime import datetime
import parsedatetime as pdt

class Intent:
    def __init__(self, intent_data: dict):
        self.__intent_data = intent_data
        self.__entities = Intent.parse_entities({e["entity"]: e["value"] for e in self.__intent_data["entities"]})

    @property
    def most_probable_intent(self) -> dict:
        return self.__intent_data["intent"]

    @property
    def all_probable_intents(self) -> list:
        return self.__intent_data["intent_ranking"]

    @property
    def entities(self) -> dict:
        return self.__entities

    """ Parses entity values to structured data if required """
    def parse_entities(entity_data: dict):
        entities = entity_data.keys()

        if "TME" in entities:
            entity_data["TME"] = Intent.parse_datetime(entity_data["TME"])
        
        return entity_data
    
    """ Parses text to structured time"""
    def parse_datetime(time_text: str) -> datetime:
        pdt_calendar = pdt.Calendar()
        dto = datetime(*(pdt_calendar.parse(time_text)[0])[:6])
        return [dto.year, dto.month, dto.day, dto.hour, dto.minute, dto.second]