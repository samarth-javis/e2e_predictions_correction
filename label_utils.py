import os
import pandas as pd
from PIL import ImageDraw, ImageFont

id2label_PATH = "."

def iob_to_label(label):

    label_parts = label.split('-')
    if len(label_parts)>1:
        return label_parts[-1]
    else:
        return label


def filter_out_of_page_elements(page_ocr_data):

    page_ocr_data = page_ocr_data[(page_ocr_data['minx']>=0) & (page_ocr_data['miny']>=0) & 
                                  (page_ocr_data['maxx']<=1) & (page_ocr_data['maxy']<=1)]
    
    return page_ocr_data


def filter_zero_width_height_elements(page_ocr_data):
    # For 20226811734211500, 202262211921390577
    page_ocr_data = page_ocr_data[(page_ocr_data['maxx'] - page_ocr_data['minx'] > 0) & (page_ocr_data['maxy'] - page_ocr_data['miny'] > 0)]
    return page_ocr_data


def filter_misc_labels(predictions, token_bboxes, labels_to_keep=[]):

    token_bboxes = [bbox for idx, bbox in enumerate(token_bboxes) 
                    if iob_to_label(predictions[idx]) in labels_to_keep]
    predictions = [pred for pred in predictions if iob_to_label(pred) in labels_to_keep]

    return predictions, token_bboxes


def add_predictions_to_ocr_data(label_preds, row_preds, col_preds, token_bboxes, ocr_data):

    ocr_data['bbox_key_string'] = ocr_data['bboxes'].apply(lambda bbox: ','.join(list(map(str, bbox))))
    token_bbox_key_strings = [','.join(list(map(str, bbox))) for bbox in token_bboxes]
    
    ocr_data['label'] = None
    if label_preds is None:
        # in case of RowColumnHead model label predictions are None, set to "O"
        ocr_data['label'] = "O"
    else:    
        for predicted_label, token_bbox_key_string in zip(label_preds, token_bbox_key_strings):
            if ocr_data.loc[ocr_data['bbox_key_string']==token_bbox_key_string, 'label'].isnull().any():
                ocr_data.loc[ocr_data['bbox_key_string']==token_bbox_key_string, 'label'] = remove_label_suffix(predicted_label)
    
    ocr_data['row'] = None
    for row, token_bbox_key_string in zip(row_preds, token_bbox_key_strings):
        if ocr_data.loc[ocr_data['bbox_key_string']==token_bbox_key_string, 'row'].isnull().any():
            ocr_data.loc[ocr_data['bbox_key_string']==token_bbox_key_string, 'row'] = row
    
    ocr_data['col'] = None
    for col, token_bbox_key_string in zip(col_preds, token_bbox_key_strings):
        if ocr_data.loc[ocr_data['bbox_key_string']==token_bbox_key_string, 'col'].isnull().any():
            ocr_data.loc[ocr_data['bbox_key_string']==token_bbox_key_string, 'col'] = col

    # Sanity Check - Filtering out words without corresponding predictions
    ocr_data = ocr_data[ocr_data['label'].notnull() & ocr_data['row'].notnull() & ocr_data['col'].notnull()]

    return ocr_data


COLOURS_LIST = {}
def get_label_colour(label_name):

    colour_idx = len(label_name) % len(COLOURS_LIST)
    return COLOURS_LIST[colour_idx]


def unnormalize_layoutlm_box(bbox, width, height):
    
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def filter_misc_labels_visualization(predictions, token_bboxes):

    token_bboxes = [bbox for idx, bbox in enumerate(token_bboxes) 
                    if iob_to_label(predictions[idx]) != "O"]
    predictions = [pred for pred in predictions if iob_to_label(pred) != "O"]

    return predictions, token_bboxes


def visualise_predictions(image, predictions, token_bboxes, docid, page_number, filename = None, show_o = False):

    if filename is None:
        filename = docid

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    img_width, img_height = image.size
    if not show_o:
        predictions, token_bboxes = filter_misc_labels_visualization(predictions, token_bboxes)
    token_bboxes = [unnormalize_layoutlm_box(sub_box, img_width, img_height) for idx, sub_box in enumerate(token_bboxes)]

    for predicted_label, box in zip(predictions, token_bboxes):
        predicted_label_2 = predicted_label
        predicted_label = iob_to_label(predicted_label)
        if box[2]<box[0] or box[3]<box[1]:
            if predicted_label!='other':
                print('Problematic box:', predicted_label, box)
            box[2] = box[0] + abs(box[2]-box[0])
            box[3] = box[1] + abs(box[3]-box[1])
        
        color = get_label_colour(predicted_label)
        if "col" in predicted_label and "col" in filename:
            sd = int(predicted_label.split("_")[-1])
            color = get_label_colour(range(sd))
        if "row" in predicted_label and "row" in filename:
            sd = 1 if ((predicted_label.split("-")[0])) == "B" else 2
            color = get_label_colour(range(sd))

        draw.rectangle(box, outline=color)
        draw.text((box[0]+10, box[1]-10), text=predicted_label_2, fill=color, font=font)

    image_save_dir = f'./tmp/output/{docid}'
    filename = f'{filename}_{page_number}.jpg'
    os.makedirs(image_save_dir, exist_ok=True)
    image.save(f'{image_save_dir}/{filename}')


def get_iou_row_match(row_y0, row_y1, row_2_y0, row_2_y1):
    intersection = [max(row_y0, row_2_y0), min(row_y1, row_2_y1)]
    if intersection[0] >= intersection[1]:
        return False
    iou = (intersection[1]-intersection[0])/min(row_y1-row_y0, row_2_y1-row_2_y0)
    if iou > 0.7:  # reduced to 0.70 because of 0.71 in 20233144535825235
        return True
    return False


def get_combined_rows(table_df):

    rows_ls = table_df['row_id'].unique()
    i = 0
    merged_row_ls = []
    while i < (len(rows_ls)-1):
        df = table_df[((table_df['row_id'] == rows_ls[i]) | (table_df['row_id'] == rows_ls[i+1]))]
        if all(df['label'].value_counts() == 1):
            table_df.loc[(table_df['row_id'] == rows_ls[i+1]), 'row_id'] = rows_ls[i]
            merged_row_ls.append([rows_ls[i], rows_ls[i+1]])
            i += 1
        i += 1

    return table_df, merged_row_ls


def merge_bbox(bbox_list):
    return [min(box[0] for box in bbox_list),
            min(box[1] for box in bbox_list),
            max(box[2] for box in bbox_list),
            max(box[3] for box in bbox_list)]


def update_merged_bbox(prev_bbox, next_bbox):
    prev_bbox[0] = min(prev_bbox[0], next_bbox[0])
    prev_bbox[1] = max(prev_bbox[1], next_bbox[1])
    prev_bbox[2] = min(prev_bbox[2], next_bbox[2])
    prev_bbox[3] = max(prev_bbox[3], next_bbox[3])
    return prev_bbox


def rename_label(label):
    label = str(label)
    label = label.replace('_table', '')
    label = label.replace('_line', '')
    label = label.replace('_header', '')
    return label


def remove_label_suffix(label):
    label = str(label)
    label = label.replace('_table', '')
    label = label.replace('_line', '')
    return label


def split_ocr_data(page_ocr_data, label_preds, row_preds, col_preds, token_bboxes):

    if page_ocr_data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ## add predictions to page_ocr
    page_ocr_data_with_labels = add_predictions_to_ocr_data(label_preds, row_preds, col_preds, token_bboxes, page_ocr_data)

    ## split page_ocr_data
    key_value_data = page_ocr_data_with_labels[(page_ocr_data_with_labels["label"].str.endswith("_key") | \
                                                page_ocr_data_with_labels["label"].str.endswith("_key_name") | \
                                                page_ocr_data_with_labels["label"].str.endswith("_header"))]
    table_data = page_ocr_data_with_labels[((page_ocr_data_with_labels["row"]!="O") & \
                                            (page_ocr_data_with_labels["col"]!="O"))]
    non_table_data = page_ocr_data_with_labels[~(
        (
            page_ocr_data_with_labels["label"].str.endswith("_key") | \
            page_ocr_data_with_labels["label"].str.endswith("_key_name") | \
            page_ocr_data_with_labels["label"].str.endswith("_header")
        ) | \
        (
            (page_ocr_data_with_labels["row"]!="O") & \
            (page_ocr_data_with_labels["col"]!="O") 
        ) | \
        (
            (page_ocr_data_with_labels["label"]=="O")
        )
    )]
    
    return key_value_data, table_data, non_table_data, page_ocr_data_with_labels
