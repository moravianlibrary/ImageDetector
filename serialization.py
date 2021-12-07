from xml.etree.ElementTree import Element, SubElement, tostring
import json


def make_json(boxes, page_width, page_height, file_identifier):
    return json.dumps({"images": [[int(i) for i in box] for box in boxes],
                       "page_height": int(page_height),
                       "page_width": int(page_width),
                       "file_identifier": file_identifier
                       }
                      )


def add_boxes_to_page(page_element, boxes):
    for x1, y1, x2, y2 in boxes:
        SubElement(page_element, "Illustration", attrib={
                   "WIDTH": str(int(x2 - x1)), "HEIGHT": str(int(y2 - y1)), "HPOS": str(int(x1)), "VPOS": str(int(y1))})


def make_alto(boxes, page_width, page_height, file_identifier):
    """simple alto containing data"""

    alto = Element("alto", attrib={"xmlns": "http://www.loc.gov/standards/alto/ns-v2#",
                                   "xmlns:xlink": "http://www.w3.org/1999/xlink", "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance"})

    description = SubElement(alto, "Description")
    SubElement(description, "MeasurementUnit").text = "pixel"
    source_image_information = SubElement(
        description, "sourceImageInformation")
    SubElement(source_image_information,
               "fileIdentifier").text = file_identifier

    layout = SubElement(alto, "Layout")

    page = SubElement(layout, "Page", attrib={
        "ID": "Page1", "PHYSICAL_IMG_NR": "1", "HEIGHT": str(int(page_height)), "WIDTH": str(int(page_width))})

    add_boxes_to_page(page, boxes)

    return tostring(alto)
