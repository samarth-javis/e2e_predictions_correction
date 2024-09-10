from utils1 import add_predictions_to_ocr_data
import pandas as pd
from utils import find_intersection_smaller, timeit
import numpy as np


def rename_label(label):
    label = str(label)
    label = label.replace('_table', '')
    label = label.replace('_line', '')
    label = label.replace('_header', '')
    return label


def iob_to_label(label):

    label_parts = label.split('-')
    if len(label_parts)>1:
        return label_parts[-1]
    else:
        return label


def get_col_num_from_col_label(col_label):
    col_number = int(str(col_label).split("_")[-1])

    return col_number


def assign_columns(table):

    if table.empty:
        table["col_num"] = table["col"]
        return table
    
    table["col_num"] = table["col"].apply(lambda x: get_col_num_from_col_label(x))
    table = table.sort_values("minx").groupby("col_num").apply(lambda x: x)
    table = table.reset_index(drop=True)
    # get indices of first occurances of unique column numbers
    indices = table.groupby("col_num").head(1).index
    # Set fac = 10 for elements where column number cyclicaly repeats, indicated by x coordinate diff greater than threshold
    table["fac"] = 10*(table.groupby("col_num")["minx"].diff()>0.15)
    # set fac = 0 for first occurances of unique column numbers for sanity
    table["fac"].loc[indices] = 0
    # Perform cummulative sum on fac to distinctly identify second third,.. cyclic repetition for each column number label
    # assignment will be 10 for second 20 for third and so on
    table["fac"]= table.groupby("col_num")["fac"].cumsum()
    table = table.reset_index(drop=True)
    # Assign final sequential column numbers as (col_num_label + fac)
    table["col_num"] = table.apply(lambda x: x["col_num"]+x["fac"], axis=1)

    table = table.sort_values("col_num")
    same_columns_separated = table[table["col_num"].diff()==10]["col_num"]
    for col_num in same_columns_separated.unique():
        table.loc[table[table["col_num"]==col_num].index, "col_num"] = col_num - 10
    table["col_num"] = table["col_num"].diff().fillna(0).cumsum().astype(int) ## ordered column number, hope this matches with_raw_df column order
    table = table.sort_values("original_order")

    return table


def recursive_split_table_seperators_finder(table_seperators, page_ocr, seperation_coord = "minx", common_label_set = set()):
    """ Returns table is table_split, value to be ignored (label_unique set), verified table_seperators """
    if len(table_seperators) == 0 :
        return False, list(set((map(lambda x: rename_label(iob_to_label(x)), page_ocr["label"].unique())))), []
    page_ocr_left = page_ocr[page_ocr[seperation_coord] < table_seperators[0]]
    page_ocr_right = page_ocr[page_ocr[seperation_coord] >= table_seperators[0]]
    ## find unique labels in the current table split/section
    label_set_left = page_ocr_left["label"].unique()
    label_set_left = (set((map(lambda x: rename_label(iob_to_label(x)), label_set_left))))
    ## combined with all label_left till now, excluding the O label
    label_set_left = label_set_left.union(common_label_set) - {"O"}
    is_right_split_table, label_set_right, table_seperators_verified = recursive_split_table_seperators_finder(table_seperators[1:], 
                                                                                                               page_ocr_right, 
                                                                                                               seperation_coord=seperation_coord, 
                                                                                                               common_label_set=label_set_left)
    intersection_left_right = set(label_set_right).intersection(label_set_left)
    percentage_intersection_left = 100*len(intersection_left_right)/len(label_set_left) if len(label_set_left) else 0
    intersection_right_left = set(label_set_left).intersection(label_set_right)
    percentage_intersection_right = 100*len(intersection_right_left)/len(label_set_right) if len(label_set_right) else 0
    split_table = is_right_split_table
    ## if both left and right table have significant columns common then we have a split.
    if percentage_intersection_left>50 and percentage_intersection_right>50:
        table_seperators_verified = [table_seperators[0]] + table_seperators_verified
        split_table = True

    return split_table, label_set_left, table_seperators_verified



def assign_level_ids(page_ocr_data):

    page_ocr_data = page_ocr_data.sort_values("miny")
    page_ocr_data["next_ymin"] = page_ocr_data["miny"].shift(-1)
    page_ocr_data["prev_ymax"] = page_ocr_data["maxy"].shift(1)
    page_ocr_data["height"] = page_ocr_data["maxy"] - page_ocr_data["miny"]
    page_ocr_data["row_breakpoints"] = (((page_ocr_data["prev_ymax"] - page_ocr_data["miny"] + page_ocr_data["height"]/2.5)<(page_ocr_data["maxy"]- page_ocr_data["next_ymin"])) | (page_ocr_data["miny"]>page_ocr_data["prev_ymax"]))#.fillna(True)
    page_ocr_data["level_id"] = page_ocr_data["row_breakpoints"].cumsum()
    page_ocr_data = page_ocr_data.sort_values(["minx","miny"])
    page_ocr_data["next_xmin"] = page_ocr_data["minx"].shift(-1)
    page_ocr_data["prev_xmax"] = page_ocr_data["maxx"].shift(1)
    page_ocr_data["width"] = page_ocr_data["maxx"] - page_ocr_data["minx"]
    page_ocr_data["value"] = (page_ocr_data["prev_xmax"] - page_ocr_data["minx"])
    page_ocr_data["value2"] = -10#page_ocr_data["width"]/2
    page_ocr_data["value"] = page_ocr_data[["value", "value2"]].max(axis=1)
    page_ocr_data["row_breakpoints"] = ((page_ocr_data["value"] + page_ocr_data["width"]/2.5)<(page_ocr_data["maxx"]- page_ocr_data["next_xmin"])) | (page_ocr_data["minx"]>page_ocr_data["prev_xmax"])
    page_ocr_data["level_id_y"] = page_ocr_data["row_breakpoints"].cumsum()
    page_ocr_data = page_ocr_data.sort_values(["level_id","level_id_y"])
    page_ocr_data = page_ocr_data.drop(columns = ["row_breakpoints", "width", "next_xmin", "next_ymin", "prev_ymax", "height", "prev_xmax"])

    return page_ocr_data


def remove_multi_line_complex_headers_split(page_ocr_data_without_headers, separators):

    verified_separators = [separators[0]]
    prev_sep = separators[0]
    for sep in separators[1::]:
        ocr_between_header_separators =page_ocr_data_without_headers[(page_ocr_data_without_headers["midy"]>prev_sep) & (page_ocr_data_without_headers["midy"]<sep)]
        prev_sep = sep
        if ocr_between_header_separators.empty:
            continue
        else:
            verified_separators.append(sep)
    
    return verified_separators


def give_y_seperators(page_ocr_data):
    ### detect verticle_seperators:
    page_ocr_data = page_ocr_data.sort_values(["level_id_y","level_id"])
    y_seperators = list(page_ocr_data[page_ocr_data["label"].str.endswith("_header")]["miny"].unique())
    header_ocr = page_ocr_data[page_ocr_data["label"].str.endswith("_header")]
    ## verify recursive y_seperators
    is_split, _, y_seperators = recursive_split_table_seperators_finder(y_seperators, header_ocr, seperation_coord = "miny")
    y_seperators = [0] + y_seperators + [2]
    if is_split:
        y_seperators = remove_multi_line_complex_headers_split(page_ocr_data[page_ocr_data["is_header_token"]==False], y_seperators)
    return y_seperators


def give_ocr_split(page_ocr_data, seperators, coordinate = "X"):
    if str(coordinate).lower()=="x":
        coordinate = "maxx"
    else:
        coordinate = "maxy"
    page_ocr_list = []
    for idx, sep in enumerate(seperators):
        if idx == 0:
            continue
        prev_sep = seperators[idx-1]
        page_ocr_data_i = page_ocr_data[(page_ocr_data[coordinate]<sep) & (page_ocr_data[coordinate]>=prev_sep)]
        if page_ocr_data_i.empty:
            continue
        page_ocr_list.append(page_ocr_data_i)
    return page_ocr_list


def give_seperator_X(page_ocr_data):
    # row and labels with B- seperation split at duplicates
    # Currently capable of handling only 2 side by side tables
    page_ocr_data["row-label"]  = page_ocr_data["row_num"].astype(str) +"-"+ page_ocr_data["label"]
    page_ocr_data = page_ocr_data[page_ocr_data["label2"]!="O"] ## check repitition at only non "O" labels
    sep_x = 2
    for row in sorted(page_ocr_data["row_num"].unique()):
        row_data = page_ocr_data[page_ocr_data["row_num"]==row].sort_values("level_id_y")
        duplicated_starting = row_data[~(row_data["row-label"].duplicated(keep="last")) & (row_data["label"]!="O")]
        row_data["cond"] = ~(row_data["row-label"].duplicated(keep="first"))
        duplicated_starting = row_data[~row_data.loc[::-1,"cond"].cumsum()[::-1].astype(bool)]
        initial_string = row_data[row_data.loc[::-1,"cond"].cumsum()[::-1].astype(bool)]
        beginner_labels = set(initial_string["label2"].unique())
        duplicated_labels = set(duplicated_starting["label2"].unique())
        percentage_intersection = 100*len(beginner_labels.intersection(duplicated_labels))/len(beginner_labels) if len(beginner_labels) else 0
        if duplicated_starting.empty or len(row_data["row-label"].unique())==1 or percentage_intersection<50:
            continue
        ## ignore the first label, reminents of last column of left table ->only if last column had reminant that is word does not contains "Ġ"
        duplicated_starting = duplicated_starting[duplicated_starting["row-label"]!=duplicated_starting["row-label"].iloc[0]] if "Ġ" not in duplicated_starting["word"].iloc[0] else duplicated_starting
        sep_x=min(sep_x,duplicated_starting["minx"].min())
    return sep_x 


def give_x_seperators(page_ocr_data):
    ## seperate and repeat
    x_seperators = []
    x_sep = give_seperator_X(page_ocr_data)
    prev_xsep = 0
    iteration = 0
    while x_sep<=1 and iteration<5:
        x_seperators.append(x_sep)
        page_ocr_data_i = page_ocr_data[(page_ocr_data["minx"]>=x_sep)]
        x_sep = give_seperator_X(page_ocr_data_i)
        iteration+=1
    ## verify:
    is_split, _, x_seperators = recursive_split_table_seperators_finder(x_seperators, page_ocr_data)
    x_seperators = [0] + x_seperators + [2]
    return x_seperators


def filter_misslabels(page_ocr_data):
    page_ocr_data["label2"] = page_ocr_data["label"].apply(lambda x: rename_label(iob_to_label(x)))
    group = page_ocr_data.groupby("label2")
    cols = group['level_id_y'].sum().index
    # avg_level_y = {str(col): int(group['level_id_y'].sum().loc[col])*int(group.agg({"level_id_y": lambda x: 1/pd.Series(x).value_counts().sum()})).loc[col] for col in cols}
    avg_level_y = {str(col): (group['level_id_y'].sum().loc[col])*(group.agg({"level_id_y": lambda x: 1/pd.Series(x).value_counts().sum()}))["level_id_y"].loc[col] for col in cols}
    distance_threshold = round(pd.Series(sorted(avg_level_y.values())).diff().fillna(0).mean()) + 10
    unconfident_tokens_group = group["level_id_y"].value_counts()[group["level_id_y"].value_counts()<3]
    for lab in page_ocr_data["label2"].unique():
        try:
            level_ys = unconfident_tokens_group[lab].index
        except:
            continue
        for lev_y in level_ys:
            if abs(lev_y - avg_level_y[lab])>distance_threshold:
                page_ocr_data.loc[(page_ocr_data["label2"]==lab) & (page_ocr_data["level_id_y"]==lev_y), "label"] = "O"
                page_ocr_data.loc[(page_ocr_data["label2"]==lab) & (page_ocr_data["level_id_y"]==lev_y), "label2"] = "O"
    return page_ocr_data


def give_col_seperators(page_ocr_data):
    col_starters = 0
    col_enders = 0
    page_ocr_data["label2"] = page_ocr_data["label"].apply(lambda x: rename_label(iob_to_label(x)))
    mean_diff = 0

    col_enders = page_ocr_data.groupby("label2")["maxx"].max()
    col_starters = page_ocr_data.groupby("label2")["minx"].min()

    col_starters = sorted(list(col_starters))
    col_enders = sorted(list(col_enders))
    col_sep = [0]
    while len(col_starters) or len(col_enders):
        while len(col_starters):
            x = col_starters.pop(0)
            if x<col_sep[-1]-0.05:
                pass
            else:
                col_sep.append(x)
                break
        while len(col_enders):
            x = col_enders.pop(0)
            if x<col_sep[-1]-0.05:
                pass
            else:
                col_sep.append(x)
                break
    return col_sep[1:]


def assign_cols(page_ocr_data, col_seperators):
    page_ocr_data["col_num"] = pd.NA
    col_seperators_min = pd.Series(col_seperators)
    col_seperators_max = col_seperators_min.shift(-1).fillna(1)
    filt_cols = range(1,len(col_seperators_max),2)
    for level_y in page_ocr_data["level_id_y"].unique():
        level_col_filter = page_ocr_data["level_id_y"]==level_y
        level_col = page_ocr_data[level_col_filter]
        mini_col = level_col["minx"].min()
        maxi_col = level_col["maxx"].max()
        col_seperators_max - mini_col
        col_seperators_max - col_seperators_min - ((maxi_col - col_seperators_min) + (col_seperators_max - mini_col))
        iou_series = ((((maxi_col - col_seperators_min).apply(lambda x: x if x>0 else 0)) - (mini_col - col_seperators_min).apply(lambda x: x if x>0 else 0) - (maxi_col- col_seperators_max).apply(lambda x: x if x>0 else 0) + (mini_col- col_seperators_max).apply(lambda x: x if x>0 else 0))/(maxi_col-mini_col+1))
        col_num = iou_series.idxmax()
        page_ocr_data.loc[level_col_filter,"col_num"] = col_num
    page_ocr_data = page_ocr_data.sort_values(["row_num","level_id_y"])
    page_ocr_data["col_num"] = page_ocr_data["col_num"].fillna(method="ffill").fillna(0)
    return page_ocr_data


def assign_rows_helper(table):
    """provides initial estimate of row ids for table"""
    table = table.sort_values(["midy", "minx"])
    table.reset_index(drop=True, inplace=True)

    B_row_indices_reversed = table[table["row"].apply(lambda x: str(x).startswith("B-"))].index[::-1]
    total_num_B_rows = len(B_row_indices_reversed)

    unable_to_match_b_rows = False
    row_difference_threshold = 0.005
    table["diff_row"] = table["midy"].diff()
    for i in range(100):
        table["row_num"] = (table["diff_row"]>row_difference_threshold).cumsum()

        # row_indices signify actual row seperators
        row_indices = list(table[table["diff_row"]>row_difference_threshold].index)
        row_indices.insert(0,0)
        if len(row_indices) < total_num_B_rows:
            row_difference_threshold = row_difference_threshold*0.8
            table["diff_row"] = table["midy"].diff()
            if len(table[table["diff_row"]>row_difference_threshold].index) > 2*total_num_B_rows or i>20: ## threshold reduction dramatically increases possible rows or we have already iterated a lot, consider assigning some rows than keep trying
                unable_to_match_b_rows = True
            else:
                continue

        # Assinging a row seperator to a cooresponding B-row seperator, based on y-coordinate distance
        row_seperation_indices = []
        for idx, b_row_i in enumerate(B_row_indices_reversed):
            y_distance_and_index_list = [[abs(table["miny"].loc[r_row_i] - table["miny"].loc[b_row_i]) , idr] \
                                         for idr, r_row_i in enumerate(row_indices)]
            nearest_row_index = sorted(y_distance_and_index_list)[0][1]
            # poping ou the nearest index so it does not get reassigned to another B-row
            row_seperation_index = row_indices.pop(nearest_row_index)
            row_seperation_indices.append(row_seperation_index)
            if len(row_indices) == 0: ## should algorithmically occur only if in last iteration of loop or unable_to_match_b_rows == True
                break

        if len(row_seperation_indices) == total_num_B_rows or unable_to_match_b_rows:
            table["row_num"] = (table.index.isin(row_seperation_indices)).cumsum()
            break

    table = table.sort_values("row_num")
    table["row_num"] = table["row_num"].diff().fillna(0).cumsum().astype(int) ## row_num should be 0-N for further processing and row_alignment check

    table = table.sort_values("original_order")

    return table


def reasure_row_nums(page_ocr_data):

    contained_labels_master = page_ocr_data["label"].unique()
    contained_labels_master = (set((map(lambda x: rename_label(iob_to_label(x)), contained_labels_master)))) - {"O"}

    for j in range(3):
        prev_miny = 1
        prev_maxy = 1
        prev_midy_dis = 0
        prev_level_id = 1000
        prev_row_id = 1000
        prev_iou = 1
        for level_id in sorted(page_ocr_data["level_id"].unique(), reverse=True):
            level_line = page_ocr_data[page_ocr_data["level_id"] == level_id]
            miny = level_line["miny"].min()
            maxy = level_line["maxy"].max()
            row_id = level_line["row_num"].iloc[0]
            iou = find_intersection_smaller(miny, maxy, prev_miny, prev_maxy)
            midy_dis = (prev_maxy+prev_miny)/2 - (miny+maxy)/2

        
            if iou>30:
                page_ocr_data.loc[page_ocr_data["level_id"] == level_id, "row_num"] = prev_row_id

            prev_maxy = maxy
            prev_miny = miny
            prev_midy_dis = midy_dis
            prev_level_id = level_id
            prev_row_id = row_id
            prev_iou = iou
        
        prev_miny = 1
        prev_maxy = 1
        prev_midy_dis = 0
        prev_level_id = 1000
        prev_row_id = 1000
        prev_iou = 1
        prev_percentage_intersection = 0
        prev_contained_labels_level = set()
        for level_id in sorted(page_ocr_data["level_id"].unique()):
            level_line = page_ocr_data[page_ocr_data["level_id"] == level_id]
            miny = level_line["miny"].min()
            maxy = level_line["maxy"].max()
            row_id = level_line["row_num"].iloc[0]
            iou = find_intersection_smaller(miny, maxy, prev_miny, prev_maxy)
            midy_dis = abs((prev_maxy+prev_miny)/2 - (miny+maxy)/2)



            contained_labels_level = level_line["label"].unique()
            contained_labels_level = (set((map(lambda x: rename_label(iob_to_label(x)), contained_labels_level)))) - {"O"}
            intersection_labels = set(contained_labels_master).intersection(contained_labels_level)
            percentage_intersection = 100*len(intersection_labels)/len(contained_labels_master) if len(contained_labels_master) else 0

            intersection_with_prev_labels = set(prev_contained_labels_level).intersection(contained_labels_level)
            percentage_intersection_with_prev = 100*len(intersection_with_prev_labels)/len(contained_labels_level) if len(contained_labels_level) else 0

            intersection_union = set(prev_contained_labels_level).union(contained_labels_level)
            percentage_intersection_union = 100*len(intersection_union)/len(contained_labels_master) if len(contained_labels_master) else 0
        
            
            if (prev_iou<=0 and  midy_dis<prev_midy_dis and prev_percentage_intersection<35 and percentage_intersection_with_prev<60 and  percentage_intersection_union>35 and percentage_intersection<30):
                page_ocr_data.loc[page_ocr_data["level_id"] == prev_level_id, "row_num"] = row_id 

            prev_maxy = maxy
            prev_miny = miny
            prev_midy_dis = midy_dis
            prev_level_id = level_id
            prev_row_id = row_id
            prev_iou = iou
            prev_contained_labels_level = contained_labels_level
            prev_percentage_intersection = percentage_intersection

    page_ocr_data = page_ocr_data.sort_values("original_order")
    
    return page_ocr_data


def assign_rows(table):
    """Assigns final row_ids over initial estimates, makes sure that 1 complete line belongs to single row"""
    table = assign_rows_helper(table)

    ## making sure that each line has unique row id
    contained_labels_master = table["label"].unique()
    contained_labels_master = (set((map(lambda x: rename_label(iob_to_label(x)), contained_labels_master)))) - {"O"}
    for level_id in table["level_id"].unique():
        level_line = table[table["level_id"] == level_id]
        row_id = level_line["row_num"].value_counts().idxmax()
        contained_labels_level = level_line["label"].unique()
        contained_labels_level = (set((map(lambda x: rename_label(iob_to_label(x)), contained_labels_level)))) - {"O"}
        intersection_labels = set(contained_labels_master).intersection(contained_labels_level)
        percentage_intersection = 100*len(intersection_labels)/len(contained_labels_master) if len(contained_labels_master) else 0
        if level_line[level_line["row"]!="I-row"].empty and percentage_intersection>50:
            row_id += 12000*level_id
        table.loc[level_line.index, "row_num"] = row_id
    table = table.sort_values("level_id")
    table["row_num"] = table["row_num"].diff().astype(bool).cumsum()


    table = reasure_row_nums(table)
    table = table.sort_values("level_id")

    table["row_num"] = table["row_num"].diff().astype(bool).cumsum()

    table = table.sort_values("original_order")

    return table


def get_row_labels_from_row_num(page_ocr_data_with_row_num):

    page_ocr_data_with_row_num = page_ocr_data_with_row_num.sort_values(["midy", "minx"])
    page_ocr_data_with_row_num.reset_index(drop=True, inplace=True)

    page_ocr_data_with_row_num["new_row"] = "I-row"

    B_row_indices = list(page_ocr_data_with_row_num[page_ocr_data_with_row_num["row_num"].diff().fillna(1)>0].index)

    header_row_idices = page_ocr_data_with_row_num[page_ocr_data_with_row_num["row"].apply(lambda x: "header_row" in str(x))].index
    if len(header_row_idices):
        B_header_row_index = header_row_idices[0]
        y_distance_and_index_list = [[abs(page_ocr_data_with_row_num["miny"].loc[r_row_i] - page_ocr_data_with_row_num["miny"].loc[B_header_row_index]) , idr] \
                                    for idr, r_row_i in enumerate(B_row_indices)]
        correct_B_header_idi_index = sorted(y_distance_and_index_list)[0][1]
        correct_B_header_index = B_row_indices.pop(correct_B_header_idi_index)
        page_ocr_data_with_row_num.loc[correct_B_header_index, "new_row"] = "B-header_row"

    page_ocr_data_with_row_num.loc[B_row_indices, "new_row"] = "B-row"
    page_ocr_data_with_row_num["new_row_num"] = 0
    page_ocr_data_with_row_num.loc[page_ocr_data_with_row_num["new_row"].apply(lambda x: str(x).startswith("B-")), "new_row_num"] = 1
    page_ocr_data_with_row_num["new_row_num"] = page_ocr_data_with_row_num["new_row_num"].cumsum()

    ## adding IB tag
    for new_row_num in page_ocr_data_with_row_num["new_row_num"].unique():
        row_group = page_ocr_data_with_row_num[page_ocr_data_with_row_num["new_row_num"]==new_row_num]
        if "B-header_row" in set(row_group["new_row"]):
            page_ocr_data_with_row_num.loc[row_group.index, "new_row"] = page_ocr_data_with_row_num.loc[row_group.index]["new_row"].apply(lambda x: str(x).replace("I-row", "I-header_row"))
        page_ocr_data_with_row_num.loc[row_group[row_group.sort_values(["level_id", "minx"])["level_id"].diff().fillna(0)>=1].index, "new_row"] =   page_ocr_data_with_row_num.loc[row_group[row_group.sort_values(["level_id", "minx"])["level_id"].diff().fillna(0)>=1].index]["new_row"].apply(lambda x: str(x).replace("I-","IB-"))

    page_ocr_data_with_row_num["row"] = page_ocr_data_with_row_num["new_row"]
    page_ocr_data_with_row_num.drop(columns = ["new_row", "row_num"])

    return page_ocr_data_with_row_num


def assign_rows_labels(page_ocr_data):

    page_ocr_data["row_id"] = 0
    # page_ocr_data["row_num"] = page_ocr_data["level_id"]
    contained_labels_master = page_ocr_data["label"].unique()
    contained_labels_master = (set((map(lambda x: rename_label(iob_to_label(x)), contained_labels_master)))) - {"O"}
    page_ocr_data = page_ocr_data.sort_values("level_id").reset_index(drop=True)
    row_id = 0
    for level_id in sorted(page_ocr_data["level_id"].unique()):
        level_line = page_ocr_data[page_ocr_data["level_id"] == level_id]
        # row_id = level_line["row_num"].value_counts().idxmax()
        contained_labels_level = level_line["label"].unique()
        contained_labels_level = (set((map(lambda x: rename_label(iob_to_label(x)), contained_labels_level)))) - {"O"}
        intersection_labels = set(contained_labels_master).intersection(contained_labels_level)
        percentage_intersection = 100*len(intersection_labels)/len(contained_labels_master) if len(contained_labels_master) else 0
        if percentage_intersection>30:
            row_id += 1
        page_ocr_data.loc[level_line.index, "row_id"] = page_ocr_data.loc[level_line.index, "row_id"]  + row_id
        # if prev_level_id have some overlap with this new row
    page_ocr_data = page_ocr_data.sort_values("row_id")
    page_ocr_data["row_num"] = page_ocr_data["row_id"].diff().astype(bool).cumsum()

    return page_ocr_data

#summarize the below function
# this function is used in assign_rows_labels, takes page_ocr_data with column number and compute corrected column numbers
# using label predictions by the following algorithm
# 1. sort by col_num
# 2. group by col_num
# 3. sort by minx
# 4. assign group number to each col_num
# 5. assign sequential group number to each col_num
# 6. assign sequential col_num to each col_num
# 7. 
def get_corrected_col_labels_from_num(page_ocr_data_with_row_col_num):

    column_table = page_ocr_data_with_row_col_num.groupby("col_num").agg({
        "col_num":"first",
        "minx": min,
        "maxx": max
        }).sort_values("minx")
    
    grp = 0
    column_table["group"] = grp
    last_maxx = 0
    for idx in column_table.index:
        curr_minx, curr_maxx = column_table.loc[idx][["minx", "maxx"]]
        if curr_minx>last_maxx:
            grp += 1
        column_table.loc[idx, "group"] = grp
        last_maxx = max(last_maxx, curr_maxx)

    total_num_rows = len(page_ocr_data_with_row_col_num["row_num"].unique())
    col_to_assign = 0
    prev_labels_maximus = "O"
    majority_to_avoid = "O"
    prev_assigned_label = "O"
    page_ocr_data_with_row_col_num["old-col_num"] = page_ocr_data_with_row_col_num["col_num"]
    for combined_group in column_table["group"].unique():
        group = column_table[column_table["group"]==combined_group]

        # check if complex column
        # this check will only verify that the major columns (defined as occurance at par with number of rows) have rythmic occorance
        is_complex_column_valid = False
        min_x_group = group["minx"].min()
        max_x_group = group["maxx"].max()
        page_ocr_within_possible_complex_column = page_ocr_data_with_row_col_num[(page_ocr_data_with_row_col_num["midx"]>=min_x_group) & ((page_ocr_data_with_row_col_num["midx"])<= max_x_group)]
        is_rythmic = {}
        major_col_num = []
        for col_num in group["col_num"].unique():
            col_within_complex_column = page_ocr_within_possible_complex_column[page_ocr_within_possible_complex_column["col_num"]==col_num].copy()
            col_within_complex_column = assign_level_ids(col_within_complex_column)

            col_rows = list(col_within_complex_column["level_id"].unique())
            if len(col_rows)<total_num_rows-3:
                continue # not a major column 
            
            major_col_num.append(col_num)

            range_rows = pd.Series(list(range(max(col_rows)+1))) if len(col_rows) else pd.Series([0])
            range_rows = range_rows*0
            range_rows.loc[col_rows] = 1

            fft = np.fft.fft(range_rows)/len(range_rows)
            fft_ = fft*0
            fft_[fft>0.8] = fft[fft>0.8]

            if sum(fft_)>=1:
                # is rythmic"
                is_rythmic[col_num] = True
            else:
                is_rythmic[col_num] = False

            ifft = np.fft.ifft(fft_*len(range_rows))

        if all([is_rythmic[col_num] for col_num in major_col_num]):
            is_complex_column_valid = True
        
        non_major_cols = set(group["col_num"].unique()) - set(major_col_num)

        needs_correction = True if len(non_major_cols) else False

        if is_complex_column_valid and not needs_correction:
            for col_num in sorted(group["col_num"].unique()):
                col_to_assign += 1
                col_within_complex_column = page_ocr_within_possible_complex_column[page_ocr_within_possible_complex_column["col_num"]==col_num]
                page_ocr_data_with_row_col_num.loc[col_within_complex_column.index, "col_num"] = col_to_assign
            continue
        elif len(major_col_num)>1 and is_complex_column_valid:
            #TODO complex column correction algo
            # we will follow simple strategy for this:
            # 1, non major col nums will be assigned "O", later they will be updated with column number which has maximum bidirectinal iou (find_larger_intersection) in horizontal axis
            # 2. columns will be mapped to their high probability label, and the map is inverted, later it is made sure that each label is mapped to unique column, high quality label predictions are assumed
            # 3, 
            col_within_complex_column = page_ocr_within_possible_complex_column[page_ocr_within_possible_complex_column["col_num"]==col_num]

            if needs_correction:
                non_major_cols_ocr = page_ocr_within_possible_complex_column[page_ocr_within_possible_complex_column["col_num"].apply(lambda x: x not in major_col_num)]
                major_cols_ocr = page_ocr_within_possible_complex_column[page_ocr_within_possible_complex_column["col_num"].apply(lambda x: x in major_col_num)]
                non_major_cols_ocr["col_num"] = "O"
                for idx_nc in non_major_cols_ocr.index:
                    minx_nc, maxx_nc = non_major_cols_ocr.loc[idx_nc][["minx", "maxx"]]
                    min1 = major_cols_ocr["minx"].apply(lambda x: min(x, minx_nc))
                    max2 = major_cols_ocr["maxx"].apply(lambda x: max(x, maxx_nc))
                    diff = major_cols_ocr["maxx"] - major_cols_ocr["minx"]
                    filter_remove = max2 > min1
                    intersection = max2-min1
                    max_width = diff.apply(lambda x: max(x, maxx_nc - minx_nc))#min((a2-a1), (b2-b1))
                    iou = intersection / max_width * 100
                    iou_index = iou[filter_remove].idxmax() if len(iou[filter_remove]) else 0
                    col_num_to_assign = major_cols_ocr.loc[iou_index, "col_num"]
                    non_major_cols_ocr.loc[idx_nc, "col_num"] = col_num_to_assign
                page_ocr_data_with_row_col_num.loc[non_major_cols_ocr.index, "col_num"] = non_major_cols_ocr["col_num"]

            #label to column homogenity is aligned regardless
            major_label_to_col_map = {}
            for col_num in major_col_num:
                col_within_complex_column = page_ocr_within_possible_complex_column[page_ocr_within_possible_complex_column["col_num"]==col_num]
                label_col = col_within_complex_column['label2'].value_counts().idxmax()
                major_label_to_col_map[label_col] = col_num
            for lab_col in major_label_to_col_map:
                lab_within_complex_column = page_ocr_within_possible_complex_column[page_ocr_within_possible_complex_column["col_num"]==col_num]
                page_ocr_data_with_row_col_num.loc[lab_within_complex_column.index, "col_num"] = major_label_to_col_map[lab_col]

            # finally update/assign new col_nums as procedure of the this function
            cols__ = [col_num_ for _, col_num_ in major_label_to_col_map.items()]
            for col_num in sorted(cols__):
                col_to_assign += 1
                col_within_complex_column = page_ocr_within_possible_complex_column[page_ocr_within_possible_complex_column["col_num"]==col_num]
                page_ocr_data_with_row_col_num.loc[col_within_complex_column.index, "col_num"] = col_to_assign

        else:
            # either single major column or is not valid complex column of does not need correction, follow routine column correction
            pass
        
        # routine column correction
        col_seps = sorted(group["minx"].to_list() + group["maxx"].to_list())

        # iteration from last column
        prev_col_sep = col_seps[0]
        prev_col_num_detected = 0
        for col_sep in col_seps[1:]:
            page_ocr_within_possible_column = page_ocr_data_with_row_col_num[(page_ocr_data_with_row_col_num["midx"]>=prev_col_sep) & ((page_ocr_data_with_row_col_num["midx"])<= col_sep)]
            page_ocr_within_possible_column_without_header = page_ocr_within_possible_column[~page_ocr_within_possible_column["row"].apply(lambda x: "header" in str(x))]
            
            if not page_ocr_within_possible_column.empty and page_ocr_within_possible_column_without_header.empty:
                col_to_assign += 1
                page_ocr_data_with_row_col_num.loc[page_ocr_within_possible_column.index, "col_num"] = col_to_assign
                prev_col_sep = col_sep
                continue
            if page_ocr_within_possible_column.empty:
                continue

            all_cols = page_ocr_within_possible_column_without_header["col"]
            all_labels = page_ocr_within_possible_column_without_header["label"]
            col_nums_detected = page_ocr_within_possible_column_without_header["old-col_num"]
            col_count_frame = col_nums_detected.value_counts()
            major_col_num = col_count_frame.idxmax()
            
            if all_labels.empty:
                continue
            all_labels = pd.Series(list((map(lambda x: rename_label(iob_to_label(x)), all_labels))))
            unique_labels = list(all_labels.unique())
            labels_maximus = all_labels[all_labels!="O"].value_counts().idxmax() if unique_labels!=["O"] and len(unique_labels) else "O"
            labels_maximus_without_prev = all_labels[(all_labels!="O") & (all_labels!=prev_labels_maximus) & (all_labels!=labels_maximus)].value_counts().idxmax() if len(set(unique_labels)-{prev_labels_maximus, labels_maximus, "O"}) else "O"
            percent_minority = 100*all_labels.value_counts()[labels_maximus_without_prev]/all_labels.value_counts().sum() if labels_maximus_without_prev!="O" else 0
            prev_labels_maximus_percent = 100*all_labels.value_counts()[prev_labels_maximus]/all_labels.value_counts().sum() if prev_labels_maximus!="O" and prev_labels_maximus in unique_labels else 0
            
            if major_col_num!=prev_col_num_detected and 100*col_count_frame[major_col_num]/col_count_frame.sum()>95:
                col_to_assign += 1
                page_ocr_data_with_row_col_num.loc[page_ocr_within_possible_column.index, "col_num"] = col_to_assign
                prev_labels_maximus = labels_maximus
                prev_assigned_label = labels_maximus_without_prev if prev_labels_maximus==labels_maximus else labels_maximus
                prev_col_sep = col_sep
                prev_col_num_detected = major_col_num
                continue
            if len(all_cols.unique()) == 2 and all_cols.value_counts().min()<=2 or len(all_cols.unique()) == 2 and len(set(unique_labels) - {"O"})<=1 :
                # skip as erroneous column marked
                if labels_maximus!=prev_labels_maximus:
                    col_to_assign += 1
                page_ocr_data_with_row_col_num.loc[page_ocr_within_possible_column.index, "col_num"] = col_to_assign
                prev_labels_maximus = labels_maximus
                prev_assigned_label = labels_maximus_without_prev if prev_labels_maximus==labels_maximus else labels_maximus
                prev_col_sep = col_sep
                prev_col_num_detected = major_col_num
                continue
            if labels_maximus==prev_labels_maximus and percent_minority<30 or prev_labels_maximus_percent>70 and prev_assigned_label==labels_maximus_without_prev:
                # assume they are the same column
                page_ocr_data_with_row_col_num.loc[page_ocr_within_possible_column.index, "col_num"] = col_to_assign
                prev_labels_maximus = labels_maximus
                prev_assigned_label = labels_maximus_without_prev if prev_labels_maximus==labels_maximus else labels_maximus
                prev_col_sep = col_sep
                prev_col_num_detected = major_col_num
                continue
            
            # else we have new column
            col_to_assign += 1
            page_ocr_data_with_row_col_num.loc[page_ocr_within_possible_column.index, "col_num"] = col_to_assign
            prev_labels_maximus = labels_maximus
            prev_assigned_label = labels_maximus_without_prev if prev_labels_maximus==labels_maximus else labels_maximus
            prev_col_sep = col_sep
            prev_col_num_detected = major_col_num

    
    page_ocr_data_with_row_col_num = page_ocr_data_with_row_col_num.sort_values("col_num")
    page_ocr_data_with_row_col_num["col_num"] = page_ocr_data_with_row_col_num["col_num"].diff().fillna(0).astype(bool).astype(int).cumsum()
    page_ocr_data_with_row_col_num["new_col"] = page_ocr_data_with_row_col_num["col_num"].apply(lambda x: "I-col_"+str(int(x)%10))
    page_ocr_data_with_row_col_num = page_ocr_data_with_row_col_num.groupby("new_col").apply(lambda x: x).reset_index(drop=True)
    page_ocr_data_with_row_col_num["new_col"] = page_ocr_data_with_row_col_num["col_num"].apply(lambda x: "I-col_"+str(int(x)%10))
    
    ## Add B- and IB- tags where needed (replace I- tags)
    for col in page_ocr_data_with_row_col_num["col_num"].unique():
        col_group = page_ocr_data_with_row_col_num[page_ocr_data_with_row_col_num["col_num"]==col]
        col_group = col_group.sort_values(["new_row_num","level_id", "minx"])
        level_change_index = col_group[col_group["level_id"].diff().fillna(1)>=1].index
        page_ocr_data_with_row_col_num.loc[level_change_index, "new_col"] = page_ocr_data_with_row_col_num.loc[level_change_index, "new_col"].apply(lambda x: str(x).replace("I-", "B-"))
        level_change_index = col_group[col_group["level_id"].diff().fillna(0)>=1].index
        page_ocr_data_with_row_col_num.loc[level_change_index, "new_col"] = page_ocr_data_with_row_col_num.loc[level_change_index, "new_col"].apply(lambda x: str(x).replace("B-", "IB-"))
        row_change_index = col_group[col_group["new_row_num"].diff().fillna(1)>=1].index
        page_ocr_data_with_row_col_num.loc[row_change_index, "new_col"] = page_ocr_data_with_row_col_num.loc[row_change_index, "new_col"].apply(lambda x: str(x).replace("I-", "B-").replace("IB-", "B-"))
        

    page_ocr_data_with_row_col_num["col"] = page_ocr_data_with_row_col_num["new_col"]

    page_ocr_data = page_ocr_data_with_row_col_num.drop(columns = ["new_row_num", "new_col"])

    return page_ocr_data


def assign_fuzz_labels_helper(page_ocr_data_without_headers):

    prev_major_label = "O"
    prev_column_number = 0
    page_ocr_data_without_headers["fuzz_labels"] = "O"
    prev_assigning_label = "O"
    seen_labels = []
    ## recomputing level_ids without headers
    page_ocr_data_without_headers = assign_level_ids(page_ocr_data_without_headers)
    for level_id_y in sorted(page_ocr_data_without_headers["level_id_y"].unique()):
        # just concerns with non header labels
        level_y_ocr_complete = page_ocr_data_without_headers[(page_ocr_data_without_headers["level_id_y"]==level_id_y)]
        if level_y_ocr_complete.empty:
            continue
        
        for col_num in level_y_ocr_complete["col_num"].unique(): ## looping over each column number separately to handle complex column cases
            level_y_ocr_col = level_y_ocr_complete[level_y_ocr_complete["col_num"]==col_num]
            unique_labels = level_y_ocr_col["label2"].unique()
            most_occuring_label = level_y_ocr_col[level_y_ocr_col["label2"]!="O"]["label2"].value_counts().idxmax() if len(set(unique_labels) - {"O"})>0 else "O"
            count_frame = level_y_ocr_col["label2"].value_counts()
            count = count_frame[most_occuring_label] if most_occuring_label in count_frame else 0
            if count<3 and count_frame.sum()>6:# ignore 1 or 2 misprediction
                most_occuring_label = "O"

            column_num = level_y_ocr_col["col_num"].value_counts().idxmax()
            assigning_label = most_occuring_label
            if "O" in count_frame and 100*count_frame["O"]/count_frame.sum()>50:
                most_occuring_label = "O"
                assigning_label = most_occuring_label
                page_ocr_data_without_headers.loc[level_y_ocr_col.index, "fuzz_labels"] = assigning_label
                prev_assigning_label = assigning_label
                prev_major_label = most_occuring_label
                prev_column_number = column_num
                continue
            if True:
                second_most_occuring_label = level_y_ocr_col[(level_y_ocr_col["label2"]!="O") & (level_y_ocr_col["label2"]!=most_occuring_label)]["label2"].value_counts()
                second_most_occuring_label = second_most_occuring_label.idxmax() if len(second_most_occuring_label) else "O"
                if most_occuring_label in count_frame and 100*count_frame[most_occuring_label]/count_frame.sum()<90 and (second_most_occuring_label!="O" and most_occuring_label == prev_major_label or most_occuring_label in seen_labels) :
                    assigning_label = second_most_occuring_label

            if most_occuring_label!=prev_major_label or assigning_label!=prev_major_label:
                seen_labels.append(prev_major_label)
            
            if len(seen_labels)>3:
                seen_labels.pop(0)
            if not (prev_column_number==column_num and prev_assigning_label!=assigning_label and 100*count_frame[assigning_label]/count_frame.sum()<75):
                page_ocr_data_without_headers.loc[level_y_ocr_col.index, "fuzz_labels"] = assigning_label
                prev_assigning_label = assigning_label
            else:
                page_ocr_data_without_headers.loc[level_y_ocr_col.index, "fuzz_labels"] = prev_assigning_label

            prev_major_label = most_occuring_label
            prev_column_number = column_num

    # update header tokens
    page_ocr_data_without_headers["label"] = page_ocr_data_without_headers["fuzz_labels"]

    return page_ocr_data_without_headers


def assign_fuzz_labels(page_ocr_data):

    page_ocr_data_without_headers = page_ocr_data[page_ocr_data["is_header_token"]==False]
    page_ocr_data_without_headers = assign_fuzz_labels_helper(page_ocr_data_without_headers)
    page_ocr_data["label"] = page_ocr_data["label2"]
    page_ocr_data.loc[page_ocr_data_without_headers.index, "label"] = page_ocr_data_without_headers["label"]

    return page_ocr_data


def IB_tag_labels(page_ocr_data):

    page_ocr_data
    for row_num in page_ocr_data["row_num"].unique():
        row_elements = page_ocr_data[page_ocr_data["row_num"]==row_num]
        if "header" in row_elements["row"].iloc[0]:
            row_elements["label"] = row_elements["label"].apply(lambda x: "I-"+str(x)+"_header")
        else:
            row_elements["label"] = row_elements["label"].apply(lambda x: "I-"+str(x))
        row_elements = row_elements.sort_values(["midy", "minx"])
        # row_elements.groupby("label").
        row_elements.loc[~row_elements["label"].duplicated(keep="first"), "label"] = row_elements[~row_elements["label"].duplicated(keep="first")]["label"].apply(lambda x: str(x).replace("I-", "B-"))
        for unique_label in row_elements["label2"].unique():
            row_unique_label = row_elements[row_elements["label2"]==unique_label]
            row_unique_label.loc[row_unique_label["level_id"].diff()>=1, "label"] = row_unique_label[row_unique_label["level_id"].diff()>=1]["label"].apply(lambda x: str(x).replace("I-", "IB-"))
            row_elements.loc[row_unique_label.index, "label"] = row_unique_label["label"]
        # row_elements.loc[row_elements["level_id"].diff()>=1, "label"] = row_elements[row_elements["level_id"].diff()>=1]["label"].apply(lambda x: str(x).replace("I-", "IB-"))
        row_elements.loc[row_elements["label2"]=="O", "label"] = "O"
        page_ocr_data.loc[row_elements.index, "label"] = row_elements["label"]
    
    return page_ocr_data



def correct_page_ocr_model_predictions(label_preds, row_preds, col_preds, token_bboxes, page_ocr_data):

    if page_ocr_data.empty:
        return label_preds, row_preds, col_preds, token_bboxes, page_ocr_data

    page_ocr_data_with_labels = add_predictions_to_ocr_data(label_preds, row_preds, col_preds, token_bboxes, page_ocr_data)
    page_ocr_data = page_ocr_data_with_labels[((page_ocr_data_with_labels["row"]!="O") & \
                                                (page_ocr_data_with_labels["col"]!="O"))]
    non_table_data = page_ocr_data_with_labels[~((page_ocr_data_with_labels["row"]!="O") & \
                                                (page_ocr_data_with_labels["col"]!="O"))]
    
    if page_ocr_data.empty:
        page_ocr_data = pd.concat([non_table_data, page_ocr_data])
        page_ocr_data = page_ocr_data.sort_values("original_order").reset_index(drop=True)
        return label_preds, row_preds, col_preds, token_bboxes, page_ocr_data

    non_table_data["row"] = "O"
    non_table_data["col"] = "O"
    
    B_row_indices = page_ocr_data_with_labels[page_ocr_data_with_labels["row"].apply(lambda x: str(x).startswith("B-"))]
    total_b_rows = len(B_row_indices)
    b_rows_survived = len(page_ocr_data[page_ocr_data["row"].apply(lambda x: str(x).startswith("B-"))])

    page_ocr_data["original_row"] = page_ocr_data["row"]
    page_ocr_data["is_header_token"] = page_ocr_data["row"].apply(lambda x: "header" in str(x)) | page_ocr_data["label"].apply(lambda x: "header" in str(x))

    final_ocr_list = []
    page_ocr_data = assign_level_ids(page_ocr_data)
    y_seperators = give_y_seperators(page_ocr_data)
    page_ocr_list_ys = give_ocr_split(page_ocr_data, y_seperators, coordinate = "y")
    for page_ocr_data_yi in page_ocr_list_ys:
        page_ocr_data_yi = page_ocr_data_yi.sort_values(["level_id","level_id_y"])
        page_ocr_data_yi = assign_rows_labels(page_ocr_data_yi)
        page_ocr_data_yi["label2"] = page_ocr_data_yi["label"].apply(lambda x: rename_label(iob_to_label(x)))
        page_ocr_data_yi_without_headers = page_ocr_data_yi[page_ocr_data_yi["is_header_token"]==False]
        x_seperators = give_x_seperators(page_ocr_data_yi_without_headers) # Calculating separators only on column values
        page_ocr_list_xs = give_ocr_split(page_ocr_data_yi, x_seperators, coordinate = "X")
        for page_ocr_data_xi in page_ocr_list_xs:
            page_ocr_data_xi = page_ocr_data_xi.reset_index(drop=True)
            ## Assume now on that we have a single table
            page_ocr_data_xi = assign_level_ids(page_ocr_data_xi)

            ## Correct B-rows predictions
            page_ocr_data_with_row_num = assign_rows(page_ocr_data_xi) if b_rows_survived>=total_b_rows and b_rows_survived>0 else \
                                         assign_rows_labels(page_ocr_data_xi)
            page_ocr_data_with_new_row_num = get_row_labels_from_row_num(page_ocr_data_with_row_num)

            ## Correct Column predictions
            page_ocr_data_with_row_col_num = assign_columns(page_ocr_data_with_new_row_num)
            page_ocr_data_xi = get_corrected_col_labels_from_num(page_ocr_data_with_row_col_num)

            ## Create Fuzz labels for non header tokens
            page_ocr_data_xi = assign_fuzz_labels(page_ocr_data_xi)

            page_ocr_data_xi = IB_tag_labels(page_ocr_data_xi)

            page_ocr_data_xi = page_ocr_data_xi.drop(columns = ["label2", "level_id", "level_id_y", "row_id", "row-label", "row_num", "col_num"], errors='ignore')
            final_ocr_list.append(page_ocr_data_xi)
        
    page_ocr_data = pd.concat(final_ocr_list).reset_index(drop=True)
    page_ocr_data = page_ocr_data.sort_values("original_order")
    page_ocr_data["row"] = page_ocr_data["original_row"]
    page_ocr_data["is_header_token"] = page_ocr_data["row"].apply(lambda x: "header" in str(x)) | page_ocr_data["label"].apply(lambda x: "header" in str(x))

    ## recalculating row for erroneous split detection
    page_ocr_data = assign_level_ids(page_ocr_data)
    y_seperators = give_y_seperators(page_ocr_data)
    if len(y_seperators)>1:
        final_ocr_list = []
        page_ocr_list_ys = give_ocr_split(page_ocr_data, y_seperators, coordinate = "y")
        for page_ocr_data_yi in page_ocr_list_ys:
            page_ocr_data_yi = page_ocr_data_yi.sort_values(["level_id","level_id_y"])
            page_ocr_data_yi = assign_rows_labels(page_ocr_data_yi)
            page_ocr_data_yi["label2"] = page_ocr_data_yi["label"].apply(lambda x: rename_label(iob_to_label(x)))
            page_ocr_data_yi_without_headers = page_ocr_data_yi[page_ocr_data_yi["is_header_token"]==False]
            x_seperators = give_x_seperators(page_ocr_data_yi_without_headers) # only on column values
            page_ocr_list_xs = give_ocr_split(page_ocr_data_yi, x_seperators, coordinate = "X")
            for page_ocr_data_xi in page_ocr_list_xs:
                page_ocr_data_xi = page_ocr_data_xi.reset_index(drop=True)
                ## Assume now on that we have a single table
                page_ocr_data_xi = assign_level_ids(page_ocr_data_xi)

                ## Correct B-rows predictions
                page_ocr_data_with_row_num = assign_rows(page_ocr_data_xi) if b_rows_survived>=total_b_rows and b_rows_survived>0 else \
                                            assign_rows_labels(page_ocr_data_xi)
                page_ocr_data_xi = get_row_labels_from_row_num(page_ocr_data_with_row_num)

                page_ocr_data_xi = page_ocr_data_xi.drop(columns = ["label2", "level_id", "level_id_y", "row_id", "row-label", "row_num"], errors='ignore')
                final_ocr_list.append(page_ocr_data_xi)
            
        page_ocr_data = pd.concat(final_ocr_list)
    page_ocr_data = pd.concat([non_table_data, page_ocr_data])

    page_ocr_data = page_ocr_data.sort_values("original_order").reset_index(drop=True)

    label_preds = page_ocr_data["label"].to_list()
    row_preds = page_ocr_data["row"].to_list()
    col_preds = page_ocr_data["col"].to_list()
    token_bboxes = page_ocr_data['bbox_key_string'].apply(lambda x: list(map(int,str(x).split(",")))).to_list()

    page_ocr_data = page_ocr_data.drop(columns=["label", "row", "col", "bbox_key_string", "is_header_token"])

    return label_preds, row_preds, col_preds, token_bboxes, page_ocr_data
