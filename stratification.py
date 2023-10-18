"""
Python (3) script to do a stratified, random selection from respondents to random mail out

Copyright (C) 2019-2021 Brett Hennig bsh [AT] sortitionfoundation.org & Paul Gölz pgoelz (AT) cs.cmu.edu & Dimitrios Lazarou 

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see
<https://www.gnu.org/licenses>.

Additional permission under GNU GPL version 3 section 7

"""
import codecs
import copy
import csv
import random
import typing
from copy import deepcopy
from io import StringIO
from math import log
from pathlib import Path
from typing import Dict, List, Tuple, FrozenSet, Iterable, Optional, Set

import cvxpy as cp
import gspread
import mip
import numpy as np
import toml
from oauth2client.service_account import ServiceAccountCredentials

# 0 means no debug message, higher number (could) mean more messages
debug = 0
# numerical deviation accepted as equality when dealing with solvers
EPS = 0.0005  # TODO: Find good value
EPS_NASH = 0.1
EPS2 = 0.00000001

DEFAULT_SETTINGS = """
# #####################################################################
#
# IF YOU EDIT THIS FILE YOU NEED TO RESTART THE APPLICATION
#
# #####################################################################

# this is written in TOML - https://github.com/toml-lang/toml

id_column = "nationbuilder_id"

# if check_same_address is true, then no 2 people from the same address will be selected
# the comparison is between TWO fields listed here, which MUST also be below in columns_to_keep
check_same_address = true
check_same_address_columns = [
    "primary_address1",
    "primary_zip"
]
max_attempts = 100
columns_to_keep = [
    "first_name",
    "last_name",
    "mobile_number",
    "email",
    "primary_address1",
    "primary_address2",
    "primary_city",
    "zip_royal_mail",
    "tag_list",
    "age",
    "gender"
]

# selection_algorithm can either be "legacy", "maximin", "leximin", or "nash"
selection_algorithm = "leximin"
"""


class NoSettingsFile(Exception):
    pass


class Settings:
    def __init__(self, id_column, columns_to_keep, check_same_address, check_same_address_columns, max_attempts,
                 selection_algorithm, json_file_path):
        try:
            assert (isinstance(id_column, str))
            assert (isinstance(columns_to_keep, list))
            # if they have no personal data this could actually be empty
            # assert(len(columns_to_keep) > 0)
            for column in columns_to_keep:
                assert (isinstance(column, str))
            assert (isinstance(check_same_address, bool))
            assert (isinstance(check_same_address_columns, list))
            # this could be empty
            assert(len(check_same_address_columns) == 2 or len(check_same_address_columns) == 0)
            for column in check_same_address_columns:
                assert (isinstance(column, str))
            assert (isinstance(max_attempts, int))
            assert (selection_algorithm in ["legacy", "maximin", "nash"])
        except AssertionError as error:
            print(error)

        self.id_column = id_column
        self.columns_to_keep = columns_to_keep
        self.check_same_address = check_same_address
        self.check_same_address_columns = check_same_address_columns
        self.max_attempts = max_attempts
        self.selection_algorithm = selection_algorithm
        self.json_file_path = json_file_path

    @classmethod
    def load_from_file(cls):
        message = ""
        settings_file_path = Path.home() / "sf_stratification_settings.toml"
        if not settings_file_path.is_file():
            with open(settings_file_path, "w", encoding='utf-8') as settings_file:
                settings_file.write(DEFAULT_SETTINGS)
            message = "Wrote default settings to '{}' - if editing is required, restart this app.".format(
                settings_file_path.absolute()
            )
        with open(settings_file_path, "r", encoding='utf-8') as settings_file:
            settings = toml.load(settings_file)
        # you can't check an address if there is no info about which columns to check...
        if len(settings['check_same_address_columns']) == 0 and settings['check_same_address'] == True:
        	message += "\nWARNING: in sf_stratification_settings.toml file check_same_address is TRUE but there are no columns listed to check! FIX THIS and RESTART this program!"
        settings['json_file_path'] = Path.home() / "secret_do_not_commit.json"
        return cls(
            settings['id_column'],
            settings['columns_to_keep'],
            settings['check_same_address'],
            settings['check_same_address_columns'],
            settings['max_attempts'],
            settings['selection_algorithm'],
            settings['json_file_path']
        ), message


# class for throwing error/fail exceptions
class SelectionError(Exception):
    def __init__(self, message):
        self.msg = message


###################################
#
# The PeopleAndCats classes below hold all the people and category info sourced from (and written to) the relevant place
#
# categories is a dict of dicts of dicts... like:
#   categories = { 'gender' : gender, 'age' : age, 'geo' : geo, 'socio' : socio }
# with each category a dict of possible values with set data, like:
#     gender = { 'Gender: Male' : { 'min' : 20, 'max' : 24, 'selected' : 0, 'remaining' : 0 },
#                'Gender: Female' : { 'min' : 21, 'max' : 25, 'selected' : 0, 'remaining' : 0 }
# etc         }
#
###################################

class PeopleAndCats():
    # Warning: "name" value is hardcoded somewhere below :-)
    category_file_field_names = ["category", "name", "min", "max"]

    def __init__(self):
        # mins and maxs (from category data) for number of people one can select
        self.min_max_people = {}
        self.original_categories = None
        self.categories_after_people = None
        self.category_content_loaded = False
        self.people_content_loaded = False
        self.people = None
        self.columns_data = None
        self.people_selected = None
        self.number_people_to_select = 0
        # this, and the two functions below, are the only annoying things needed to distinguish CSV in GUI..
        self.enable_file_download = False

    def get_selected_file( self ):
        return None
    def get_remaining_file( self ):
        return None

    # read in categories - a dict of dicts of dicts...
    def _read_in_cats( self, cat_head, cat_body ):
        self.original_categories = {}
        # to keep track of number in cats - number people selected MUST be between these limits in every cat...
        self.min_max_people = {}
        # check that the fieldnames are (at least) what we expect, and only once!
        for fn in PeopleAndCats.category_file_field_names:
            cat_head_fn_count = cat_head.count(fn)
            if cat_head_fn_count == 0:
                raise Exception(
                    "Did not find required column name '{}' in the input (found only {}) ".format(fn, cat_head)
                )
            elif cat_head_fn_count > 1:
                raise Exception(
                    "Found MORE THAN 1 column named '{}' in the input (found {}) ".format(fn, cat_head)
                )
        for row in cat_body:
            # allow for some dirty data - at least strip white space from cat and name
            # but only if they are strings! (sometimes people use ints as cat names and then strip produces as exception...)
            cat = row["category"]
            # and skip over any blank lines...
            if cat == '':
                continue
            if isinstance(cat, str):
                cat = cat.strip()
            # check for blank entries and report a meaningful error
            cat_value = row["name"]
            if cat_value == '' or row["min"] == '' or row["max"] == '':
                raise Exception(
                    "ERROR reading in category file: found a blank cell in a row of the category: {}. ".format(cat)
                )
            if isinstance(cat_value, str):
                cat_value = cat_value.strip()
            # must convert min/max to ints
            cat_min = int(row["min"])
            cat_max = int(row["max"])
            if cat in self.original_categories:
                self.min_max_people[cat]["min"] += cat_min
                self.min_max_people[cat]["max"] += cat_max
                self.original_categories[cat].update(
                    {
                        cat_value: {
                            "min": cat_min,
                            "max": cat_max,
                            "selected": 0,
                            "remaining": 0,
                        }
                    }
                )
            else:
                self.min_max_people.update(
                    {
                        cat: {
                            "min": cat_min,
                            "max": cat_max
                        }
                    }
                )
                self.original_categories.update(
                    {
                        cat: {
                            cat_value: {
                                "min": cat_min,
                                "max": cat_max,
                                "selected": 0,
                                "remaining": 0,
                            }
                        }
                    }
                )

        msg = [ "Number of categories: {}".format(len(self.original_categories.keys())) ]

        # work out what the min and max number of people should be,
        # given these cats
        max_values = [v['max'] for v in self.min_max_people.values()]
        max_val = min(max_values)
        min_values = [v['min'] for v in self.min_max_people.values()]
        min_val = max(min_values)
        # if the min is bigger than the max we're in trouble i.e. there's an input error
        if min_val > max_val:
            raise Exception(
                "Inconsistent numbers in min and max in the categories input: the sum of the minimum values of a category is larger than the sum of the maximum values of a(nother) category. "
            )
        return msg, min_val, max_val

    # read in people and calculate how many people in each category in database
    def _init_categories_people(self, people_head, people_body, settings: Settings):
        people = {}
        columns_data = {}
        # this modifies the categories, so we keep the original categories here
        self.categories_after_people = deepcopy(self.original_categories)
        categories = self.categories_after_people
        #people_data = csv.DictReader(people_file)
        # check that id_column and all the categories and columns_to_keep are in the people data fields
        msg = []
        try:
            # check both for existence and duplicate column names
            id_column_count = people_head.count(settings.id_column)
            if id_column_count == 0:
                raise Exception(
                    "No {} (unique id) column found in people data!".format(settings.id_column)
                )
            elif id_column_count > 1:
                raise Exception(
                    "MORE THAN 1 {} (unique id) column found in people data!".format(settings.id_column)
                )
            for cat_key in categories.keys():
                cat_key_count = people_head.count(cat_key)
                if cat_key_count == 0:
                    raise Exception(
                        "No '{}' (category) column found in people data!".format(cat_key)
                    )
                elif cat_key_count > 1:
                    raise Exception(
                        "MORE THAN 1 '{}' (category) column found in people data!".format(cat_key)
                    )
            for column in settings.columns_to_keep:
                column_count = people_head.count(column)
                if column_count == 0:
                    raise Exception(
                        "No '{}' column (to keep) found in people data!".format(column)
                    )
                elif column_count > 1:
                    raise Exception(
                        "MORE THAN 1 '{}' column (to keep) found in people data!".format(column)
                    )
            for column in settings.check_same_address_columns:
                column_count = people_head.count(column)
                if column_count == 0:
                    raise Exception(
                        "No '{}' column (to check same address) found in people data!".format(column)
                    )
                elif column_count > 1:
                    raise Exception(
                        "MORE THAN 1 '{}' column (to check same address) found in people data!".format(column)
                    )
            for row in people_body:
                pkey = row[settings.id_column]
                # skip over any blank lines... but warn the user
                if pkey == '':
                    msg += [ "<b>WARNING</b>: blank cell found in ID column - skipped that line!"]
                    continue
                value = {}
                for cat_key, cats in categories.items():
                    # check for input errors here - if it's not in the list of category values...
                    # allow for some unclean data - at least strip empty space, but only if a str!
                    # (some values will can be numbers)
                    p_value = row[cat_key]
                    if isinstance(row[cat_key], str):
                        p_value = p_value.strip()
                    if p_value not in cats:
                        raise Exception(
                            "ERROR reading in people (init_categories_people): Person (id = {}) has value '{}' not in category {}".format(pkey, p_value, cat_key)
                        )
                    value.update({cat_key: p_value})
                    categories[cat_key][p_value]["remaining"] += 1
                people.update({pkey: value})
                # this is address, name etc that we need to keep for output file
                data_value = {}
                for col in settings.columns_to_keep:
                    data_value[col] = row[col]
                columns_data.update({pkey: data_value})
            # check if any cat[max] is set to zero... if so delete everyone with that cat...
            # NOT DONE: could then check if anyone is left...
            total_num_people = len(people.keys())
            msg += ["Number of people: {}.".format(total_num_people)]
            total_num_deleted = 0
            for cat_key, cats in categories.items():
                for cat, cat_item in cats.items():
                    if cat_item["max"] == 0:  # we don't want any of these people
                        num_deleted, num_left = delete_all_in_cat(categories, people, cat_key, cat)
                        total_num_deleted += num_deleted
                        msg += [ "Category {} full - deleted {}, {} left.".format(cat, num_deleted, num_left) ]
            # if the total number of people deleted is lots then we're probably doing a replacement selection, which means
            # the 'remaining' file will be useless - remind the user of this!
            if total_num_deleted > total_num_people/2:
                msg += [ ">>> WARNING <<< That deleted MANY PEOPLE - are you doing a replacement? If so your REMAINING FILE WILL BE USELESS!!!" ]
            self.people = people
            self.columns_data = columns_data
        except Exception as error:
            self.people_content_loaded = False
            msg += [ "Error loading people: {}".format(error) ]
        return msg

    def people_cats_run_stratification( self, settings: Settings, test_selection):
        # if this is being called again (the user hit the button again!) we want to make sure all data is cleared etc
        # but the function called here makes deep copies of categories_after_people and people
        self.people_selected = None
        success, self.people_selected, output_lines = run_stratification(
            self.categories_after_people, self.people, self.columns_data, self.number_people_to_select, self.min_max_people, settings, test_selection
        )
        if success:
            # this also outputs them...
            output_lines += self._get_selected_people_lists( settings )
        return success, output_lines

    # this also outputs them by calling the appropriate derived class method...
    def _get_selected_people_lists( self, settings: Settings):
        people_working = copy.deepcopy(self.people)
        people_selected = self.people_selected
        categories = self.categories_after_people
        columns_data = self.columns_data

        people_selected_rows = [ [settings.id_column] + settings.columns_to_keep + list(categories.keys()) ]
        people_remaining_rows = [ [settings.id_column] + settings.columns_to_keep + list(categories.keys()) ]

        output_lines = []  # do we want to output same address info? Nah, this overwritten at the end of this function...
        num_same_address_deleted = 0
        for pkey, person in people_selected.items():
            row = [pkey]
            for col in settings.columns_to_keep:
                row.append(columns_data[pkey][col])
            row += person.values()
            people_selected_rows += [ row ]
            # if check address then delete all those at this address (will delete the one we want as well)
            if settings.check_same_address:
                people_to_delete, new_output_lines = get_people_at_same_address(people_working, pkey, columns_data, settings.check_same_address_columns)
                output_lines += new_output_lines
                num_same_address_deleted += len(new_output_lines) - 1  # don't include original
                # then delete this/these people at the same address from the reserve/remaining pool
                for del_person_key in people_to_delete:
                    del people_working[del_person_key]
            else:
                del people_working[pkey]

        # add the columns to keep into to remaining people
        for pkey, person in people_working.items():
            row = [pkey]
            for col in settings.columns_to_keep:
                row.append(columns_data[pkey][col])
            row += person.values()
            people_remaining_rows += [ row ]
        output_lines = ["Deleted {} people from remaining file who had the same address as selected people.".format(num_same_address_deleted)]
        self._output_selected_remaining( settings, people_selected_rows, people_remaining_rows )
        return output_lines


class PeopleAndCatsCSV(PeopleAndCats):

    def __init__(self):
        super(PeopleAndCatsCSV, self).__init__()
        # self.people_csv_content = ''
        self.selected_file = StringIO()
        self.remaining_file = StringIO()

    def get_selected_file(self):
        return self.selected_file

    def get_remaining_file(self):
        return self.remaining_file

    def load_cats(self, file_contents, settings: Settings):
        self.category_content_loaded = True
        category_file = StringIO(file_contents)
        category_reader = csv.DictReader(category_file)
        return self._read_in_cats(list(category_reader.fieldnames), category_reader)

    def load_people(self, settings: Settings, file_contents='', dummy_respondents_tab='', dummy_gen_rem=''):
        if file_contents != '':
            self.people_content_loaded = True
        people_file = StringIO(file_contents)
        people_data = csv.DictReader(people_file)
        return self._init_categories_people(list(people_data.fieldnames), people_data, settings)

    # Actually useful to also write to a file all those who are NOT selected for later selection if people pull out etc
    # BUT, we should not include in this people from the same address as someone who has been selected!
    def _output_selected_remaining(self, settings: Settings, people_selected_rows, people_remaining_rows):
        # we have succeeded in CSV so can activate buttons in GUI...
        self.enable_file_download = True

        people_selected_writer = csv.writer(
            self.selected_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for row in people_selected_rows:
            people_selected_writer.writerow(row)

        people_remaining_writer = csv.writer(
            self.remaining_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for row in people_remaining_rows:
            people_remaining_writer.writerow(row)


class PeopleAndCatsGoogleSheet(PeopleAndCats):
    scope = None
    creds = None
    client = None
    category_tab_name = "Categories"
 #   respondents_tab_name = "Respondents"##Nick is taking this out
    original_selected_tab_name = "Original Selected - output - "
    selected_tab_name = "Selected"
    columns_selected_first = "C"
    column_selected_blank_num = 6
    remaining_tab_name = "Remaining - output - "
    new_tab_default_size_rows = "2"
    new_tab_default_size_cols = "40"

    def __init__(self):
        super(PeopleAndCatsGoogleSheet, self).__init__()
        self.g_sheet_name = ''
        self.respondents_tab_name = ''
        self.gen_rem_tab='' 
        self.spreadsheet = None

    def _tab_exists(self, tab_name):
        if self.spreadsheet == None:
            return False
        tab_list = self.spreadsheet.worksheets()
        for tab in tab_list:
            if tab.title == tab_name:
                return True
        return False

    def _clear_or_create_tab(self, tab_name, other_tab_name, inc):
        # this now does not clear data but increments the sheet number...
        num = 0
        tab_ready = None
        tab_name_new = tab_name + str(num)
        other_tab_name_new = other_tab_name + str(num)
        while tab_ready == None:
            if self._tab_exists(tab_name_new) or self._tab_exists(other_tab_name_new):
                num += 1
                tab_name_new = tab_name + str(num)
                other_tab_name_new = other_tab_name + str(num)
            # tab_ready = self.spreadsheet.worksheet(tab_name )
            # tab_ready.clear()
            else:
                if inc == -1:
                    tab_name_new = tab_name + str(num-1)
                tab_ready = self.spreadsheet.add_worksheet(title=tab_name_new, rows=self.new_tab_default_size_rows,
                                                           cols=self.new_tab_default_size_cols)
        return tab_ready

    def load_cats(self, g_sheet_name, settings: Settings):
        self.category_content_loaded = True
        self.g_sheet_name = g_sheet_name

        json_file_name = settings.json_file_path
        min_val = 0
        max_val = 0
        msg = []
        try:
            if self.scope is None:
                self.scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
                self.creds = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, self.scope)
                self.client = gspread.authorize(self.creds)
            self.spreadsheet = self.client.open(self.g_sheet_name)
            msg += ["Opened Google Sheet: '{}'. ".format(self.g_sheet_name)]
            if self._tab_exists(self.category_tab_name):
                tab_cats = self.spreadsheet.worksheet(self.category_tab_name)
                cat_head_input = tab_cats.row_values(1)
                cat_input = tab_cats.get_all_records()
                new_msg, min_val, max_val = self._read_in_cats(cat_head_input, cat_input)
                msg += ["Read in '{}' tab in above Google sheet.".format(self.category_tab_name)]
                msg += new_msg
            else:
                msg += ["Error in Google sheet: no tab called '{}' found. ".format(self.category_tab_name)]
                self.category_content_loaded = False
        except gspread.SpreadsheetNotFound:
            msg += ["Google spreadsheet not found: {}. ".format(self.g_sheet_name)]
            self.category_content_loaded = False
        return msg, min_val, max_val

##Added respondents_tab_name and gen_rem_tab as an argument
    def load_people(self, settings: Settings, dummy_file_contents, respondents_tab_name, gen_rem_tab):
        self.people_content_loaded = True
        self.respondents_tab_name = respondents_tab_name ##Added for respondents tab text box.
        self.gen_rem_tab = gen_rem_tab ##Added for checkbox
        try:
            if self._tab_exists(self.respondents_tab_name):
                tab_people = self.spreadsheet.worksheet(self.respondents_tab_name)
                # if we don't read this in here we can't check if there are 2 columns with the same name
                people_head_input = tab_people.row_values(1)
                # the numericise_ignore doesn't convert the phone numbers to ints...
                people_input = tab_people.get_all_records(numericise_ignore=['all'])
                msg = ["Reading in '{}' tab in above Google sheet.".format(self.respondents_tab_name)]
                msg += self._init_categories_people(people_head_input, people_input, settings)
            else:
                msg = ["Error in Google sheet: no tab called '{}' found. ".format(self.respondents_tab_name)]
                self.people_content_loaded = False
        except gspread.SpreadsheetNotFound:
            msg += ["Google spreadsheet not found: {}. ".format(self.g_sheet_name)]
            self.people_content_loaded = False
        return msg

## The if statement is new.
    def _output_selected_remaining(self, settings: Settings, people_selected_rows, people_remaining_rows):

        tab_original_selected = self._clear_or_create_tab(self.original_selected_tab_name, self.remaining_tab_name,0)
        tab_original_selected.update(people_selected_rows)
        if self.gen_rem_tab=='on':
            tab_remaining = self._clear_or_create_tab(self.remaining_tab_name, self.original_selected_tab_name,-1)
            tab_remaining.update(people_remaining_rows)


###################################

# create READABLE example file of people
def create_readable_sample_file(categories, people_file: typing.TextIO, number_people_example_file, settings: Settings):
    example_people_writer = csv.writer(
        people_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    cat_keys = categories.keys()
    example_people_writer.writerow([settings.id_column] + settings.columns_to_keep + list(cat_keys))
    for x in range(number_people_example_file):
        row = ["p{}".format(x)]
        for col in settings.columns_to_keep:
            row.append(col + str(x))
        for cat_key, cats in categories.items():  # e.g. gender
            cat_items_list_weighted = []
            for cats_key, cats_item in cats.items():  # e.g. male
                for y in range(cats_item["max"]):
                    cat_items_list_weighted.append(cats_key)
            # random_cat_value = random.choice(list(cats.keys()))
            random_cat_value = random.choice(cat_items_list_weighted)
            row.append(random_cat_value)
        example_people_writer.writerow(row)


# when a category is full we want to delete everyone in it
def delete_all_in_cat(categories, people, cat, cat_value):
    people_to_delete = []
    for pkey, person in people.items():
        if person[cat] == cat_value:
            people_to_delete.append(pkey)
            for pcat, pval in person.items():
                cat_item = categories[pcat][pval]
                cat_item["remaining"] -= 1
                if cat_item["remaining"] == 0 and cat_item["selected"] < cat_item["min"]:
                    raise SelectionError(
                        "FAIL in delete_all_in_cat: no one/not enough left in " + pval
                    )
    for p in people_to_delete:
        del people[p]
    # return the number of people deleted and the number of people left
    return len(people_to_delete), len(people)


# selected = True means we are deleting because they have been chosen,
# otherwise they are being deleted because they live at same address as someone selected
def really_delete_person(categories, people, pkey, selected):
    for pcat, pval in people[pkey].items():
        cat_item = categories[pcat][pval]
        if selected:
            cat_item["selected"] += 1
        cat_item["remaining"] -= 1
        if cat_item["remaining"] == 0 and cat_item["selected"] < cat_item["min"]:
            raise SelectionError("FAIL in delete_person: no one left in " + pval)
    del people[pkey]


def get_people_at_same_address(people, pkey, columns_data, check_same_address_columns):
    # primary_address1 = columns_data[pkey]["primary_address1"]
    # primary_zip = columns_data[pkey]["primary_zip"]
    primary_address1 = columns_data[pkey][check_same_address_columns[0]]
    primary_zip = columns_data[pkey][check_same_address_columns[1]]
    # there may be multiple people to delete, and deleting them as we go gives an error
    people_to_delete = []
    output_lines = []
    for compare_key in people.keys():
        if (
                # primary_address1 == columns_data[compare_key]["primary_address1"]
                # and primary_zip == columns_data[compare_key]["primary_zip"]
                primary_address1 == columns_data[compare_key][check_same_address_columns[0]]
                and primary_zip == columns_data[compare_key][check_same_address_columns[1]]
        ):
            # found same address
            output_lines += [
                "Found someone with the same address as a selected person,"
                " so deleting him/her. Address: {} , {}".format(primary_address1, primary_zip)
            ]
            people_to_delete.append(compare_key)
    return people_to_delete, output_lines


# lucky person has been selected - delete person from DB
def delete_person(categories, people, pkey, columns_data, check_same_address, check_same_address_columns):
    output_lines = []
    # recalculate all category values that this person was in
    person = people[pkey]
    really_delete_person(categories, people, pkey, True)
    # check if there are other people at the same address - if so, remove them!
    if check_same_address:
        people_to_delete, output_lines = get_people_at_same_address(people, pkey, columns_data,
                                                                    check_same_address_columns)
        # then delete this/these people at the same address
        for del_person_key in people_to_delete:
            really_delete_person(categories, people, del_person_key, False)
    # then check if any cats of selected person is (was) in are full
    for (pcat, pval) in person.items():
        cat_item = categories[pcat][pval]
        if cat_item["selected"] == cat_item["max"]:
            num_deleted, num_left = delete_all_in_cat(categories, people, pcat, pval)
            output_lines += ["Category {} full - deleted {}, {} left.".format(pval, num_deleted, num_left)]
    return output_lines


# returns dict of category key, category item name, random person number
def find_max_ratio_cat(categories):
    ratio = -100.0
    key_max = ""
    index_max_name = ""
    random_person_num = -1
    for cat_key, cats in categories.items():
        for cat, cat_item in cats.items():
            # if there are zero remaining, or if there are less than how many we need we're in trouble
            if cat_item["selected"] < cat_item["min"] and cat_item["remaining"] < (
                    cat_item["min"] - cat_item["selected"]
            ):
                raise SelectionError(
                    "FAIL in find_max_ratio_cat: No people (or not enough) in category " + cat
                )
            # if there are none remaining, it must be because we have reached max and deleted them
            # or, if max = 0, then we don't want any of these (could happen when seeking replacements)
            if cat_item["remaining"] != 0 and cat_item["max"] != 0:
                item_ratio = (cat_item["min"] - cat_item["selected"]) / float(cat_item["remaining"])
                # print item['name'],': ', item['remaining'], 'ratio : ', item_ratio
                if item_ratio > 1:  # trouble!
                    raise SelectionError("FAIL in find_max_ratio_cat: a ratio > 1...")
                if item_ratio > ratio:
                    ratio = item_ratio
                    key_max = cat_key
                    index_max_name = cat
                    random_person_num = random.randint(1, cat_item["remaining"])
    if debug > 0:
        print("Max ratio: {} for {} {}".format(ratio, key_max, index_max_name))
        # could also append random_person_num
    return {
        "ratio_cat": key_max,
        "ratio_cat_val": index_max_name,
        "ratio_random": random_person_num,
    }


def print_category_selected(categories, number_people_wanted):
    report_msg = "<table border='1' cellpadding='5'>"
    report_msg += "<tr><th colspan='2'>Category</th><th>Selected</th><th>Want</th><th>Remaining</th></tr>"
    for cat_key, cats in categories.items():  # print out how many in each
        for cat, cat_item in cats.items():
            percent_selected = round(
                cat_item["selected"] * 100 / float(number_people_wanted), 2
            )
            report_msg += "<tr><td>{}</td><td>{}</td><td>{} ({}%)</td><td>[{},{}]</td><td>{}</td></tr>".format(
                cat_key,
                cat,
                cat_item["selected"],
                percent_selected,
                cat_item["min"],
                cat_item["max"],
                cat_item["remaining"],
            )
    report_msg += "</table>"
    return [report_msg]


def check_min_cats(categories):
    output_msg = []
    got_min = True
    for cat_key, cats in categories.items():
        for cat, cat_item in cats.items():
            if cat_item["selected"] < cat_item["min"]:
                got_min = False
                output_msg = ["Failed to get minimum in category: {}".format(cat)]
    return got_min, output_msg


def _distribution_stats(people: Dict[str, Dict[str, str]], committees: List[FrozenSet[str]],
                        probabilities: List[float]) -> List[str]:
    output_lines = []

    assert len(committees) == len(probabilities)
    num_non_zero = sum([1 for prob in probabilities if prob > 0])
    output_lines.append(f"Algorithm produced distribution over {len(committees)} committees, out of which "
                        f"{num_non_zero} are chosen with positive probability.")

    individual_probabilities = {id: 0 for id in people}
    containing_committees = {id: [] for id in people}
    for committee, prob in zip(committees, probabilities):
        if prob > 0:
            for id in committee:
                individual_probabilities[id] += prob
                containing_committees[id].append(committee)

    table = ["<table border='1' cellpadding='5'><tr><th>Agent ID</th><th>Probability of selection</th><th>Included in #"
             "of committees</th></tr>"]

    for _, id in sorted((prob, id) for id, prob in individual_probabilities.items()):
        table.append(f"<tr><td>{id}</td><td>{individual_probabilities[id]:.4%}</td><td>{len(containing_committees[id])}"
                     "</td></tr>")
    table.append("</table>")
    output_lines.append("".join(table))

    return output_lines


def _output_panel_table(panels: List[FrozenSet[str]], probs: List[float]):
    def panel_to_tuple(panel: FrozenSet[str]) -> Tuple[str]:
        return tuple(sorted(panel))

    k = len(panels[0])
    dist = {}
    for panel, prob in zip(panels, probs):
        assert len(panel) == k
        tup = panel_to_tuple(panel)
        if tup not in dist:
            dist[tup] = 0.
        dist[tup] += prob

    with codecs.open("table.csv", "w", "utf8") as file:
        file.write(",".join(["Panel number", "Suggested probability"] + [f"agent {i}" for i in range(1, k + 1)]))
        file.write("\n")
        number = 0
        for tup, prob in dist.items():
            if prob > 0:
                file.write(f"{number},{prob},")
                number += 1
                file.write(",".join(f'"{id}"' for id in tup))
                file.write("\n")


def find_random_sample(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                       columns_data: Dict[str, Dict[str, str]], number_people_wanted: int, check_same_address: bool,
                       check_same_address_columns: List[str], selection_algorithm: str, test_selection: bool) \
        -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Main algorithm to try to find a random sample.

    Args:
        categories: categories["feature"]["value"] is a dictionary with keys "min", "max", "selected", "remaining".
        people: people["nationbuilder_id"] is dictionary mapping "feature" to "value" for a person.
        columns_data: columns_data["nationbuilder_id"] is dictionary mapping "contact_field" to "value" for a person.
        number_people_wanted:
        check_same_address:
        check_same_address_columns: list of contact fields of columns that have to be equal for being
            counted as residing at the same address
        selection_algorithm: one out of "legacy", "maximin", "leximin", or "nash"
    Returns:
        (people_selected, output_lines)
        `people_selected` is a subdictionary of `people` with `number_people_wanted` many entries, guaranteed to satisfy
            the constraints on a feasible committee.
        `output_lines` is a list of debug strings.
    Raises:
        InfeasibleQuotasError if the quotas cannot be satisfied, which includes a suggestion for how to modify them.
        SelectionError in multiple other failure cases.
    Side Effects:
        Existing callers assume the "selected" and "remaining" fields in `categories` to be changed.
    """
    if check_same_address and len(check_same_address_columns) == 0:
        raise ValueError("Since the algorithm is configured to prevent multiple house members to appear on the same "
                         "panel (check_same_address = true), check_same_address_columns must not be empty.")

    # just go quick and nasty so we can hook up our charts ands tables :-)
    if test_selection:
        print("Running test selection.")
        return _find_any_committee(categories, people, columns_data, number_people_wanted, check_same_address,
                                   check_same_address_columns)

    output_lines = []
    if selection_algorithm == "leximin":
        try:
            import gurobipy
        except ModuleNotFoundError:
            output_lines.append(_print("The leximin algorithm requires the optimization library Gurobi to be installed "
                                       "(commercial, free academic licenses available). Switching to the simpler "
                                       "maximin algorithm, which can be run using open source solvers."))
            selection_algorithm = "maximin"

    if selection_algorithm == "legacy":
        return find_random_sample_legacy(categories, people, columns_data, number_people_wanted, check_same_address,
                                         check_same_address_columns)
    elif selection_algorithm == "leximin":
        committees, probabilities, new_output_lines = find_distribution_leximin(categories, people, columns_data,
                                                                                number_people_wanted,
                                                                                check_same_address,
                                                                                check_same_address_columns)
    elif selection_algorithm == "maximin":
        committees, probabilities, new_output_lines = find_distribution_maximin(categories, people, columns_data,
                                                                                number_people_wanted,
                                                                                check_same_address,
                                                                                check_same_address_columns)
    elif selection_algorithm == "nash":
        committees, probabilities, new_output_lines = find_distribution_nash(categories, people, columns_data,
                                                                             number_people_wanted, check_same_address,
                                                                             check_same_address_columns)
    else:
        # selection_algorithm not in ["legacy", "leximin", "maximin", "nash"]:
        raise ValueError(f"Unknown selection algorithm {repr(selection_algorithm)}.")

    assert len(set(committees)) == len(committees)
    #write panel output:
    _output_panel_table(committees, probabilities)
    output_lines += new_output_lines
    output_lines += _distribution_stats(people, committees, probabilities)

    # choose a concrete committee from the distribution
    committee: FrozenSet[str] = np.random.choice(list(committees), 1, p=probabilities)[0]
    people_selected = {id: people[id] for id in committee}

    # update categories for the algorithms other than legacy
    for id, person in people_selected.items():
        for feature in person:
            value = person[feature]
            categories[feature][value]["selected"] += 1
            categories[feature][value]["remaining"] -= 1
    return people_selected, output_lines


def find_random_sample_legacy(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                              columns_data: Dict[str, Dict[str, str]], number_people_wanted: int,
                              check_same_address: bool, check_same_address_columns: List[str]) \
        -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    output_lines = ["Using legacy algorithm."]
    people_selected = {}
    for count in range(number_people_wanted):
        ratio = find_max_ratio_cat(categories)
        # find randomly selected person with the category value
        for pkey, pvalue in people.items():
            if pvalue[ratio["ratio_cat"]] == ratio["ratio_cat_val"]:
                # found someone with this category value...
                ratio["ratio_random"] -= 1
                if ratio["ratio_random"] == 0:  # means they are the random one we want
                    if debug > 0:
                        print("Found random person in this cat... adding them")
                    people_selected.update({pkey: pvalue})
                    output_lines += delete_person(categories, people, pkey, columns_data, check_same_address,
                                                  check_same_address_columns)
                    break
        if count < (number_people_wanted - 1) and len(people) == 0:
            raise SelectionError("Fail! We've run out of people...")
    return people_selected, output_lines


def _ilp_results_to_committee(variables: Dict[str, mip.entities.Var]) -> FrozenSet[str]:
    try:
        res = frozenset(id for id in variables if variables[id].x > 0.5)
    except Exception as e:  # unfortunately, MIP sometimes throws generic Exceptions rather than a subclass.
        raise ValueError(f"It seems like some variables does not have a value. Original exception: {e}.")

    return res


def _same_address(columns_data1: Dict[str, str], columns_data2: Dict[str, str], check_same_address_columns: List[str]) \
        -> bool:
    return all(columns_data1[column] == columns_data2[column] for column in check_same_address_columns)


def _print(message: str) -> str:
    print(message)
    return message


def _compute_households(people: Dict[str, Dict[str, str]], columns_data: Dict[str, Dict[str, str]],
                        check_same_address_columns: List[str]) -> Dict[str, int]:
    ids = list(people.keys())
    households = {id: None for id in people}  # for each agent, the id of the earliest person with same address

    counter = 0
    for i, id1 in enumerate(ids):
        if households[id1] is not None:
            continue
        households[id1] = counter
        for id2 in ids[i + 1:]:
            if households[id2] is None and _same_address(columns_data[id1], columns_data[id2],
                                                         check_same_address_columns):
                households[id2] = counter
        counter += 1

    if counter == 1:
        print("Warning: All pool members live in the same household. Probably, the configuration is wrong?")

    return households


class InfeasibleQuotasError(Exception):
    def __init__(self, quotas: Dict[Tuple[str, str], Tuple[int, int]], output: List[str]):
        self.quotas = quotas
        self.output = ["The quotas are infeasible:"] + output

    def __str__(self):
        return "\n".join(self.output)


def _relax_infeasible_quotas(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                             number_people_wanted: int, check_same_address: bool,
                             households: Optional[Dict[str, int]] = None,
                             ensure_inclusion: typing.Collection[Iterable[str]] = ((),)) \
        -> Tuple[Dict[Tuple[str, str], Tuple[int, int]], List[str]]:
    """Assuming that the quotas are not satisfiable, suggest a minimal relaxation that would be.

    Args:
        categories: quotas in the format described in `find_random_sample`
        people: pool members in the format described in `find_random_sample`
        number_people_wanted: desired size of the panel
        check_same_address: whether members from the same household cannot simultaneously appear
        households: if `check_same_address` is given, a dictionary mapping pool member ids to integers representing
            households. if two agents have the same value in the dictionary, they are considered to live together.
        ensure_inclusion: allows to specify that some panels should contain specific sets of agents. for example,
            passing `(("a",), ("b", "c"))` means that the quotas should be relaxed such that some valid panel contains
            agent "a" and some valid panel contains both agents "b" and "c". the default of `((),)` just requires
            a panel to exist, without further restrictions.
    """
    model = mip.Model(sense=mip.MINIMIZE)
    model.verbose = debug

    assert len(ensure_inclusion) > 0  # otherwise, the existence of a panel is not required

    # for every feature, a variable for how much the upper and lower quotas are relaxed
    feature_values = [(feature, value) for feature in categories for value in categories[feature]]
    min_vars = {fv: model.add_var(var_type=mip.INTEGER, lb=0.) for fv in feature_values}
    max_vars = {fv: model.add_var(var_type=mip.INTEGER, lb=0.) for fv in feature_values}

    # we might not be able to select multiple persons from the same household
    if check_same_address:
        assert households is not None

        people_by_household = {}
        for id, household in households.items():
            if household not in people_by_household:
                people_by_household[household] = []
            people_by_household[household].append(id)

    for inclusion_set in ensure_inclusion:
        # for every person, we have a binary variable indicating whether they are in the committee
        agent_vars = {id: model.add_var(var_type=mip.BINARY) for id in people}
        for agent in inclusion_set:
            model.add_constr(agent_vars[agent] == 1)

        # we have to select exactly `number_people_wanted` many persons
        model.add_constr(mip.xsum(agent_vars.values()) == number_people_wanted)

        # we have to respect the relaxed quotas quotas
        for feature, value in feature_values:
            number_feature_value_agents = mip.xsum(agent_vars[id] for id, person in people.items()
                                                   if person[feature] == value)
            model.add_constr(
                number_feature_value_agents >= categories[feature][value]["min"] - min_vars[(feature, value)])
            model.add_constr(
                number_feature_value_agents <= categories[feature][value]["max"] + max_vars[(feature, value)])

            if check_same_address:
                for household, members in people_by_household.items():
                    if len(members) >= 2:
                        model.add_constr(mip.xsum(agent_vars[id] for id in members) <= 1)

    def reduction_weight(feature, value):
        """Make the algorithm more recluctant to reduce lower quotas that are already low. If the lower quotas was 1,
        reducing it one more (to 0) is 3 times more salient than increasing a quota by 1. This bonus tampers off
        quickly, reducing from 10 is only 1.2 times as salient as an increase."""
        old_quota = categories[feature][value]["min"]
        if old_quota == 0:
            return 0  # cannot be relaxed anyway
        else:
            return 1 + 2 / old_quota

    # we want to minimize the amount by which we have to relax quotas
    model.objective = mip.xsum(
        [reduction_weight(*fv) * min_vars[fv] for fv in feature_values] + [max_vars[fv] for fv in feature_values])

    # Optimize once without any constraints to check if no feasible committees exist at all.
    status = model.optimize()
    if status != mip.OptimizationStatus.OPTIMAL:
        raise SelectionError(f"No feasible committees found, solver returns code {status} (see "
                             f"https://docs.python-mip.com/en/latest/classes.html#optimizationstatus). Either the pool "
                             f"is very bad or something is wrong with the solver.")

    output_lines = []
    new_quotas = {}
    for fv in feature_values:
        feature, value = fv
        lower = categories[feature][value]["min"] - round(min_vars[fv].x)
        assert lower <= categories[feature][value]["min"]
        if lower < categories[feature][value]["min"]:
            output_lines.append(f"Recommend lowering lower quota of {feature}:{value} to {lower}.")
        upper = categories[feature][value]["max"] + round(max_vars[fv].x)
        assert upper >= categories[feature][value]["max"]
        if upper > categories[feature][value]["max"]:
            assert lower == categories[feature][value]["min"]
            output_lines.append(f"Recommend raising upper quota of {feature}:{value} to {upper}.")
        new_quotas[fv] = (lower, upper)

    return new_quotas, output_lines


def _setup_committee_generation(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                                number_people_wanted: int, check_same_address: bool,
                                households: Optional[Dict[str, int]]) \
        -> Tuple[mip.model.Model, Dict[str, mip.entities.Var]]:
    model = mip.Model(sense=mip.MAXIMIZE)
    model.verbose = debug

    # for every person, we have a binary variable indicating whether they are in the committee
    agent_vars = {id: model.add_var(var_type=mip.BINARY) for id in people}

    # we have to select exactly `number_people_wanted` many persons
    model.add_constr(mip.xsum(agent_vars.values()) == number_people_wanted)

    # we have to respect quotas
    for feature in categories:
        for value in categories[feature]:
            number_feature_value_agents = mip.xsum(agent_vars[id] for id, person in people.items()
                                                   if person[feature] == value)
            model.add_constr(number_feature_value_agents >= categories[feature][value]["min"])
            model.add_constr(number_feature_value_agents <= categories[feature][value]["max"])

    # we might not be able to select multiple persons from the same household
    if check_same_address:
        people_by_household = {}
        for id, household in households.items():
            if household not in people_by_household:
                people_by_household[household] = []
            people_by_household[household].append(id)

        for household, members in people_by_household.items():
            if len(members) >= 2:
                model.add_constr(mip.xsum(agent_vars[id] for id in members) <= 1)

    # Optimize once without any constraints to check if no feasible committees exist at all.
    status = model.optimize()
    if status == mip.OptimizationStatus.INFEASIBLE:
        new_quotas, output_lines = _relax_infeasible_quotas(categories, people, number_people_wanted,
                                                            check_same_address, households)
        raise InfeasibleQuotasError(new_quotas, output_lines)
    elif status != mip.OptimizationStatus.OPTIMAL:
        raise SelectionError(f"No feasible committees found, solver returns code {status} (see "
                             "https://docs.python-mip.com/en/latest/classes.html#optimizationstatus).")

    return model, agent_vars


def _find_any_committee(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                        columns_data: Dict[str, Dict[str, str]], number_people_wanted: int, check_same_address: bool,
                        check_same_address_columns: List[str]):
    if check_same_address:
        households = _compute_households(people, columns_data, check_same_address_columns)
    else:
        households = None

    model, agent_vars = _setup_committee_generation(categories, people, number_people_wanted, check_same_address,
                                                    households)
    committee = _ilp_results_to_committee(agent_vars)

    people_selected = {agent: people[agent] for agent in committee}
    for id, person in people_selected.items():
        for feature in person:
            value = person[feature]
            categories[feature][value]["selected"] += 1
            categories[feature][value]["remaining"] -= 1

    return people_selected, []


def _generate_initial_committees(new_committee_model: mip.model.Model, agent_vars: Dict[str, mip.entities.Var],
                                 multiplicative_weights_rounds: int) \
        -> Tuple[Set[FrozenSet[str]], FrozenSet[str], List[str]]:
    """To speed up the main iteration of the maximin and Nash algorithms, start from a diverse set of feasible
    committees. In particular, each agent that can be included in any committee will be included in at least one of
    these committees.
    """
    new_output_lines = []
    committees: Set[FrozenSet[str]] = set()  # Committees discovered so far
    covered_agents: Set[str] = set()  # All agents included in some committee

    # We begin using a multiplicative-weight stage. Each agent has a weight starting at 1.
    weights = {id: 1 for id in agent_vars}
    for i in range(multiplicative_weights_rounds):
        # In each round, we find a
        # feasible committee such that the sum of weights of its members is maximal.
        new_committee_model.objective = mip.xsum(weights[id] * agent_vars[id] for id in agent_vars)
        new_committee_model.optimize()
        new_set = _ilp_results_to_committee(agent_vars)

        # We then decrease the weight of each agent in the new committee by a constant factor. As a result, future
        # rounds will strongly prioritize including agents that appear in few committees.
        for id in new_set:
            weights[id] *= 0.8
        # We rescale the weights, which does not change the conceptual algorithm but prevents floating point problems.
        coefficient_sum = sum(weights.values())
        for id in agent_vars:
            weights[id] *= len(agent_vars) / coefficient_sum

        if new_set not in committees:
            # We found a new committee, and repeat.
            committees.add(new_set)
            for id in new_set:
                covered_agents.add(id)
        else:
            # If our committee is already known, make all weights a bit more equal again to mix things up a little.
            for id in agent_vars:
                weights[id] = 0.9 * weights[id] + 0.1

        print(
            f"Multiplicative weights phase, round {i + 1}/{multiplicative_weights_rounds}. Discovered {len(committees)}"
            " committees so far.")

    # If there are any agents that have not been included so far, try to find a committee including this specific agent.
    for id in agent_vars:
        if id not in covered_agents:
            new_committee_model.objective = agent_vars[id]  # only care about agent `id` being included.
            new_committee_model.optimize()
            new_set: FrozenSet[str] = _ilp_results_to_committee(agent_vars)
            if id in new_set:
                committees.add(new_set)
                for id2 in new_set:
                    covered_agents.add(id2)
            else:
                new_output_lines.append(_print(f"Agent {id} not contained in any feasible committee."))

    # We assume in this stage that the quotas are feasible.
    assert len(committees) >= 1

    if len(covered_agents) == len(agent_vars):
        new_output_lines.append(_print("All agents are contained in some feasible committee."))

    return committees, frozenset(covered_agents), new_output_lines


def _dual_leximin_stage(people: Dict[str, Dict[str, str]], committees: Set[FrozenSet[str]],
                        fixed_probabilities: Dict[str, float]):
    """This implements the dual LP described in `find_distribution_leximin`, but where P only ranges over the panels
    in `committees` rather than over all feasible panels:
    minimize ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ
    s.t.     Σ_{i ∈ P} yᵢ ≤ ŷ                              ∀ P
             Σ_{i not in fixed_probabilities} yᵢ = 1
             ŷ, yᵢ ≥ 0                                     ∀ i

    Returns a Tuple[grb.Model, Dict[str, grb.Var], grb.Var]   (not in type signature to prevent global gurobi import.)
    """
    import gurobipy as grb
    assert len(committees) != 0

    model = grb.Model()
    agent_vars = {person: model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.) for person in people}  # yᵢ
    cap_var = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.)  # ŷ
    model.addConstr(grb.quicksum(agent_vars[person] for person in people if person not in fixed_probabilities) == 1)
    for committee in committees:
        model.addConstr(grb.quicksum(agent_vars[person] for person in committee) <= cap_var)
    model.setObjective(cap_var - grb.quicksum(
        fixed_probabilities[person] * agent_vars[person] for person in fixed_probabilities),
                       grb.GRB.MINIMIZE)

    # Change Gurobi configuration to encourage strictly complementary (“inner”) solutions. These solutions will
    # typically allow to fix more probabilities per outer loop of the leximin algorithm.
    model.setParam("Method", 2)  # optimize via barrier only
    model.setParam("Crossover", 0)  # deactivate cross-over

    return model, agent_vars, cap_var


def find_distribution_leximin(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                              columns_data: Dict[str, Dict[str, str]], number_people_wanted: int,
                              check_same_address: bool, check_same_address_columns: List[str]) \
        -> Tuple[List[FrozenSet[str]], List[float], List[str]]:
    """Find a distribution over feasible committees that maximizes the minimum probability of an agent being selected
    (just like maximin), but breaks ties to maximize the second-lowest probability, breaks further ties to maximize the
    third-lowest probability and so forth.

    Arguments follow the pattern of `find_random_sample`.

    Returns:
        (committees, probabilities, output_lines)
        `committees` is a list of feasible committees, where each committee is represented by a frozen set of included
            agent ids.
        `probabilities` is a list of probabilities of equal length, describing the probability with which each committee
            should be selected.
        `output_lines` is a list of debug strings.
    """
    import gurobipy as grb

    output_lines = ["Using leximin algorithm."]
    grb.setParam("OutputFlag", 0)

    if check_same_address:
        households = _compute_households(people, columns_data, check_same_address_columns)
    else:
        households = None

    # Set up an ILP `new_committee_model` that can be used for discovering new feasible committees maximizing some
    # sum of weights over the agents.
    new_committee_model, agent_vars = _setup_committee_generation(categories, people, number_people_wanted,
                                                                  check_same_address, households)

    # Start by finding some initial committees, guaranteed to cover every agent that can be covered by some committee
    committees: Set[FrozenSet[str]]  # set of feasible committees, add more over time
    covered_agents: FrozenSet[str]  # all agent ids for agents that can actually be included
    committees, covered_agents, new_output_lines = _generate_initial_committees(new_committee_model, agent_vars,
                                                                                3 * len(people))
    output_lines += new_output_lines

    # Over the course of the algorithm, the selection probabilities of more and more agents get fixed to a certain value
    fixed_probabilities: Dict[str, float] = {}

    reduction_counter = 0

    # The outer loop maximizes the minimum of all unfixed probabilities while satisfying the fixed probabilities.
    # In each iteration, at least one more probability is fixed, but often more than one.
    while len(fixed_probabilities) < len(people):
        print(f"Fixed {len(fixed_probabilities)}/{len(people)} probabilities.")

        dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(people, committees, fixed_probabilities)
        # In the inner loop, there is a column generation for maximizing the minimum of all unfixed probabilities
        while True:
            """The primal LP being solved by column generation, with a variable x_P for each feasible panel P:
            
            maximize z
            s.t.     Σ_{P : i ∈ P} x_P ≥ z                         ∀ i not in fixed_probabilities
                     Σ_{P : i ∈ P} x_P ≥ fixed_probabilities[i]    ∀ i in fixed_probabilities
                     Σ_P x_P ≤ 1                                   (This should be thought of as equality, and wlog.
                                                                   optimal solutions have equality, but simplifies dual)
                     x_P ≥ 0                                       ∀ P
                     
            We instead solve its dual linear program:
            minimize ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ
            s.t.     Σ_{i ∈ P} yᵢ ≤ ŷ                              ∀ P
                     Σ_{i not in fixed_probabilities} yᵢ = 1
                     ŷ, yᵢ ≥ 0                                     ∀ i
            """
            dual_model.optimize()
            if dual_model.status != grb.GRB.OPTIMAL:
                # In theory, the LP is feasible in the first iterations, and we only add constraints (by fixing
                # probabilities) that preserve feasibility. Due to floating-point issues, however, it may happen that
                # Gurobi still cannot satisfy all the fixed probabilities in the primal (meaning that the dual will be
                # unbounded). In this case, we slightly relax the LP by slightly reducing all fixed probabilities.
                for agent in fixed_probabilities:
                    # Relax all fixed probabilities by a small constant
                    fixed_probabilities[agent] = max(0., fixed_probabilities[agent] - 0.0001)
                    dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(people, committees,
                                                                                    fixed_probabilities)
                print(dual_model.status, f"REDUCE PROBS for {reduction_counter}th time.")
                reduction_counter += 1
                continue

            # Find the panel P for which Σ_{i ∈ P} yᵢ is largest, i.e., for which Σ_{i ∈ P} yᵢ ≤ ŷ is tightest
            agent_weights = {person: agent_var.x for person, agent_var in dual_agent_vars.items()}
            new_committee_model.objective = mip.xsum(agent_weights[person] * agent_vars[person] for person in people)
            new_committee_model.optimize()
            new_set = _ilp_results_to_committee(agent_vars)  # panel P
            value = new_committee_model.objective_value  # Σ_{i ∈ P} yᵢ

            upper = dual_cap_var.x  # ŷ
            dual_obj = dual_model.objVal  # ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ

            output_lines.append(_print(f"Maximin is at most {dual_obj - upper + value:.2%}, can do {dual_obj:.2%} with "
                                       f"{len(committees)} committees. Gap {value - upper:.2%}."))
            if value <= upper + EPS:
                # Within numeric tolerance, the panels in `committees` are enough to constrain the dual, i.e., they are
                # enough to support an optimal primal solution.
                for person, agent_weight in agent_weights.items():
                    if agent_weight > EPS and person not in fixed_probabilities:
                        # `agent_weight` is the dual variable yᵢ of the constraint "Σ_{P : i ∈ P} x_P ≥ z" for
                        # i = `person` in the primal LP. If yᵢ is positive, this means that the constraint must be
                        # binding in all optimal solutions [1], and we can fix `person`'s probability to the
                        # optimal value of the primal/dual LP.
                        # [1] Theorem 3.3 in: Renato Pelessoni. Some remarks on the use of the strict complementarity in
                        # checking coherence and extending coherent probabilities. 1998.
                        fixed_probabilities[person] = max(0, dual_obj)
                break
            else:
                # Given that Σ_{i ∈ P} yᵢ > ŷ, the current solution to `dual_model` is not yet a solution to the dual.
                # Thus, add the constraint for panel P and recurse.
                assert new_set not in committees
                committees.add(new_set)
                dual_model.addConstr(grb.quicksum(dual_agent_vars[id] for id in new_set) <= dual_cap_var)

    # The previous algorithm computed the leximin selection probabilities of each agent and a set of panels such that
    # the selection probabilities can be obtained by randomizing over these panels. Here, such a randomization is found.
    primal = grb.Model()
    # Variables for the output probabilities of the different panels
    committee_vars = [primal.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.) for _ in committees]
    # To avoid numerical problems, we formally minimize the largest downward deviation from the fixed probabilities.
    eps = primal.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.)
    primal.addConstr(grb.quicksum(committee_vars) == 1)  # Probabilities add up to 1
    for person, prob in fixed_probabilities.items():
        person_probability = grb.quicksum(comm_var for committee, comm_var in zip(committees, committee_vars)
                                          if person in committee)
        primal.addConstr(person_probability >= prob - eps)
    primal.setObjective(eps, grb.GRB.MINIMIZE)
    primal.optimize()

    # Bound variables between 0 and 1 and renormalize, because np.random.choice is sensitive to small deviations here
    probabilities = np.array([comm_var.x for comm_var in committee_vars]).clip(0, 1)
    probabilities = list(probabilities / sum(probabilities))

    return list(committees), probabilities, output_lines


def _find_maximin_primal(committees: List[FrozenSet[str]], covered_agents: FrozenSet[str]) -> List[float]:
    model = mip.Model(sense=mip.MAXIMIZE)

    committee_variables = [model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=1.) for _ in committees]
    model.add_constr(mip.xsum(committee_variables) == 1)
    agent_panel_variables = {id: [] for id in covered_agents}
    for committee, var in zip(committees, committee_variables):
        for id in committee:
            if id in covered_agents:
                agent_panel_variables[id].append(var)

    lower = model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=1.)

    for agent_variables in agent_panel_variables.values():
        model.add_constr(lower <= mip.xsum(agent_variables))
    model.objective = lower
    model.optimize()

    probabilities = [var.x for var in committee_variables]
    probabilities = [max(p, 0) for p in probabilities]
    sum_probabilities = sum(probabilities)
    probabilities = [p / sum_probabilities for p in probabilities]
    return probabilities


def find_distribution_maximin(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                              columns_data: Dict[str, Dict[str, str]], number_people_wanted: int,
                              check_same_address: bool, check_same_address_columns: List[str]) \
        -> Tuple[List[FrozenSet[str]], List[float], List[str]]:
    """Find a distribution over feasible committees that maximizes the minimum probability of an agent being selected.

    Arguments follow the pattern of `find_random_sample`.

    Returns:
        (committees, probabilities, output_lines)
        `committees` is a list of feasible committees, where each committee is represented by a frozen set of included
            agent ids.
        `probabilities` is a list of probabilities of equal length, describing the probability with which each committee
            should be selected.
        `output_lines` is a list of debug strings.
    """
    output_lines = [_print("Using maximin algorithm.")]

    if check_same_address:
        households = _compute_households(people, columns_data, check_same_address_columns)
    else:
        households = None

    # Set up an ILP `new_committee_model` that can be used for discovering new feasible committees maximizing some
    # sum of weights over the agents.
    new_committee_model, agent_vars = _setup_committee_generation(categories, people, number_people_wanted,
                                                                  check_same_address, households)

    # Start by finding some initial committees, guaranteed to cover every agent that can be covered by some committee
    committees: Set[FrozenSet[str]]  # set of feasible committees, add more over time
    covered_agents: FrozenSet[str]  # all agent ids for agents that can actually be included
    committees, covered_agents, new_output_lines = _generate_initial_committees(new_committee_model, agent_vars,
                                                                                len(people))
    output_lines += new_output_lines

    # The incremental model is an LP with a variable y_e for each entitlement e and one more variable z.
    # For an agent i, let e(i) denote her entitlement. Then, the LP is:
    #
    # minimize  z
    # s.t.      Σ_{i ∈ B} y_{e(i)} ≤ z   ∀ feasible committees B (*)
    #           Σ_e y_e = 1
    #           y_e ≥ 0                  ∀ e
    #
    # At any point in time, constraint (*) is only enforced for the committees in `committees`. By linear-programming
    # duality, if the optimal solution with these reduced constraints satisfies all possible constraints, the committees
    # in `committees` are enough to find the maximin distribution among them.
    incremental_model = mip.Model(sense=mip.MINIMIZE)
    incremental_model.verbose = debug

    upper_bound = incremental_model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=mip.INF)  # variable z
    # variables y_e
    incr_agent_vars = {id: incremental_model.add_var(var_type=mip.CONTINUOUS, lb=0., ub=1.) for id in covered_agents}

    # Σ_e y_e = 1
    incremental_model.add_constr(mip.xsum(incr_agent_vars.values()) == 1)
    # minimize z
    incremental_model.objective = upper_bound

    for committee in committees:
        committee_sum = mip.xsum([incr_agent_vars[id] for id in committee])
        # Σ_{i ∈ B} y_{e(i)} ≤ z   ∀ B ∈ `committees`
        incremental_model.add_constr(committee_sum <= upper_bound)

    while True:
        status = incremental_model.optimize()
        assert status == mip.OptimizationStatus.OPTIMAL

        entitlement_weights = {id: incr_agent_vars[id].x for id in covered_agents}  # currently optimal values for y_e
        upper = upper_bound.x  # currently optimal value for z

        # For these fixed y_e, find the feasible committee B with maximal Σ_{i ∈ B} y_{e(i)}.
        new_committee_model.objective = mip.xsum(entitlement_weights[id] * agent_vars[id] for id in covered_agents)
        new_committee_model.optimize()
        new_set = _ilp_results_to_committee(agent_vars)
        value = sum(entitlement_weights[id] for id in new_set)

        output_lines.append(_print(f"Maximin is at most {value:.2%}, can do {upper:.2%} with {len(committees)} "
                                   f"committees. Gap {value - upper:.2%}{'≤' if value - upper <= EPS else '>'}{EPS:%}."))
        if value <= upper + EPS:
            # No feasible committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z (at least up to EPS, to prevent rounding errors).
            # Thus, we have enough committees.
            committee_list = list(committees)
            probabilities = _find_maximin_primal(committee_list, covered_agents)
            return committee_list, probabilities, output_lines
        else:
            # Some committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z. We add B to `committees` and recurse.
            assert new_set not in committees
            committees.add(new_set)
            incremental_model.add_constr(mip.xsum(incr_agent_vars[id] for id in new_set) <= upper_bound)

            # Heuristic for better speed in practice:
            # Because optimizing `incremental_model` takes a long time, we would like to get multiple committees out
            # of a single run of `incremental_model`. Rather than reoptimizing for optimal y_e and z, we find some
            # feasible values y_e and z by modifying the old solution.
            # This heuristic only adds more committees, and does not influence correctness.
            counter = 0
            for _ in range(10):
                # scale down the y_{e(i)} for i ∈ `new_set` to make Σ_{i ∈ `new_set`} y_{e(i)} ≤ z true.
                for id in new_set:
                    entitlement_weights[id] *= upper / value
                # This will change Σ_e y_e to be less than 1. We rescale the y_e and z.
                sum_weights = sum(entitlement_weights.values())
                if sum_weights < EPS:
                    break
                for id in entitlement_weights:
                    entitlement_weights[id] /= sum_weights
                upper /= sum_weights

                new_committee_model.objective = mip.xsum(entitlement_weights[id] * agent_vars[id]
                                                         for id in covered_agents)
                new_committee_model.optimize()
                new_set = _ilp_results_to_committee(agent_vars)
                value = sum(entitlement_weights[id] for id in new_set)
                if value <= upper + EPS or new_set in committees:
                    break
                else:
                    committees.add(new_set)
                    incremental_model.add_constr(mip.xsum(incr_agent_vars[id] for id in new_set) <= upper_bound)
                counter += 1
            if counter > 0:
                print(f"Heuristic successfully generated {counter} additional committees.")


def _define_entitlements(covered_agents: FrozenSet[str]) -> Tuple[List[str], Dict[str, int]]:
    entitlements = list(covered_agents)
    contributes_to_entitlement = {}
    for id in covered_agents:
        contributes_to_entitlement[id] = entitlements.index(id)

    return entitlements, contributes_to_entitlement


def _committees_to_matrix(committees: List[FrozenSet[str]], entitlements: list,
                          contributes_to_entitlement: Dict[str, int]) -> np.ndarray:
    columns = []
    for committee in committees:
        column = [0 for _ in entitlements]
        for id in committee:
            column[contributes_to_entitlement[id]] += 1
        columns.append(np.array(column))
    return np.column_stack(columns)


def find_distribution_nash(categories: Dict[str, Dict[str, Dict[str, int]]], people: Dict[str, Dict[str, str]],
                           columns_data: Dict[str, Dict[str, str]], number_people_wanted: int, check_same_address: bool,
                           check_same_address_columns: List[str]) \
        -> Tuple[List[FrozenSet[str]], List[float], List[str]]:
    """Find a distribution over feasible committees that maximizes the so-called Nash welfare, i.e., the product of
    selection probabilities over all persons.

    Arguments follow the pattern of `find_random_sample`.

    Returns:
        (committees, probabilities, output_lines)
        `committees` is a list of feasible committees, where each committee is represented by a frozen set of included
            agent ids.
        `probabilities` is a list of probabilities of equal length, describing the probability with which each committee
            should be selected.
        `output_lines` is a list of debug strings.

    The following gives more details about the algorithm:
    Instead of directly maximizing the product of selection probabilities Πᵢ pᵢ, we equivalently maximize
    log(Πᵢ pᵢ) = Σᵢ log(pᵢ). If some person/household i is not included in any feasible committee, their pᵢ is 0, and
    this sum is -∞. We will then try to maximize Σᵢ log(pᵢ) where i is restricted to range over persons/households that
    can possibly be included.
    """
    output_lines = ["Using Nash algorithm."]

    if check_same_address:
        households = _compute_households(people, columns_data, check_same_address_columns)
    else:
        households = None

    # `new_committee_model` is an integer linear program (ILP) used for discovering new feasible committees.
    # We will use it many times, putting different weights on the inclusion of different agents to find many feasible
    # committees.
    new_committee_model, agent_vars = _setup_committee_generation(categories, people, number_people_wanted,
                                                                  check_same_address, households)

    # Start by finding committees including every agent, and learn which agents cannot possibly be included.
    committees: List[FrozenSet[str]]  # set of feasible committees, add more over time
    covered_agents: FrozenSet[str]  # all agent ids for agents that can actually be included
    committee_set, covered_agents, new_output_lines = _generate_initial_committees(new_committee_model, agent_vars,
                                                                                   2 * len(people))
    committees = list(committee_set)
    output_lines += new_output_lines

    # Map the covered agents to indices in a list for easier matrix representation.
    entitlements: List[str]
    contributes_to_entitlement: Dict[str, int]  # for id of a covered agent, the corresponding index in `entitlements`
    entitlements, contributes_to_entitlement = _define_entitlements(covered_agents)

    # Now, the algorithm proceeds iteratively. First, it finds probabilities for the committees already present in
    # `committees` that maximize the sum of logarithms. Then, reusing the old ILP, it finds the feasible committee
    # (possibly outside of `committees`) such that the partial derivative of the sum of logarithms with respect to the
    # probability of outputting this committee is maximal. If this partial derivative is less than the maximal partial
    # derivative of any committee already in `committees`, the Karush-Kuhn-Tucker conditions (which are sufficient in
    # this case) imply that the distribution is optimal even with all other committees receiving probability 0.
    start_lambdas = [1 / len(committees) for _ in committees]
    while True:
        lambdas = cp.Variable(len(committees))  # probability of outputting a specific committee
        lambdas.value = start_lambdas
        # A is a binary matrix, whose (i,j)th entry indicates whether agent `feasible_agents[i]`
        matrix = _committees_to_matrix(committees, entitlements, contributes_to_entitlement)
        assert matrix.shape == (len(entitlements), len(committees))

        objective = cp.Maximize(cp.sum(cp.log(matrix * lambdas)))
        constraints = [0 <= lambdas, sum(lambdas) == 1]
        problem = cp.Problem(objective, constraints)
        # TODO: test relative performance of both solvers, see whether warm_start helps.
        try:
            nash_welfare = problem.solve(solver=cp.SCS, warm_start=True)
        except cp.SolverError:
            # At least the ECOS solver in cvxpy crashes sometimes (numerical instabilities?). In this case, try another
            # solver. But hope that SCS is more stable.
            output_lines.append(_print("Had to switch to ECOS solver."))
            nash_welfare = problem.solve(solver=cp.ECOS, warm_start=True)
        scaled_welfare = nash_welfare - len(entitlements) * log(number_people_wanted / len(entitlements))
        output_lines.append(_print(f"Scaled Nash welfare is now: {scaled_welfare}."))

        assert lambdas.value.shape == (len(committees),)
        entitled_utilities = matrix.dot(lambdas.value)
        assert entitled_utilities.shape == (len(entitlements),)
        assert (entitled_utilities > EPS2).all()
        entitled_reciprocals = 1 / entitled_utilities
        assert entitled_reciprocals.shape == (len(entitlements),)
        differentials = entitled_reciprocals.dot(matrix)
        assert differentials.shape == (len(committees),)

        obj = []
        for id in covered_agents:
            obj.append(entitled_reciprocals[contributes_to_entitlement[id]] * agent_vars[id])
        new_committee_model.objective = mip.xsum(obj)
        new_committee_model.optimize()

        new_set = _ilp_results_to_committee(agent_vars)
        value = sum(entitled_reciprocals[contributes_to_entitlement[id]] for id in new_set)
        if value <= differentials.max() + EPS_NASH:
            probabilities = np.array(lambdas.value).clip(0, 1)
            probabilities = list(probabilities / sum(probabilities))
            # TODO: filter 0-probability committees?
            return committees, probabilities, output_lines
        else:
            print(value, differentials.max(), value - differentials.max())
            assert new_set not in committees
            committees.append(new_set)
            start_lambdas = np.array(lambdas.value).resize(len(committees))


###################################
#
# main algorithm call
#
###################################


def run_stratification(categories, people, columns_data, number_people_wanted, min_max_people, settings: Settings, test_selection ):
    # First check if numbers in cat file and to select make sense
    for mkey, mvalue in min_max_people.items():
        if settings.selection_algorithm == "legacy" and (  # For other algorithms, quotas are analyzed later
                number_people_wanted < mvalue["min"] or number_people_wanted > mvalue["max"]):
            error_msg = (
                "The number of people to select ({}) is out of the range of the numbers of people "
                "in one of the {} categories. It should be within [{}, {}].".format(
                    number_people_wanted, mkey, mvalue["min"], mvalue["max"]
                )
            )
            return False, 0, {}, [error_msg]
    success = False
    tries = 0
    output_lines = []
    if test_selection:
        output_lines.append("<b style='color: red'>WARNING: Panel is not selected at random! Only use for testing!</b><br>")
    output_lines.append("<b>Initial: (selected = 0, remaining = {})</b>".format(len(people.keys())))
    while not success and tries < settings.max_attempts:
        people_selected = {}
        new_output_lines = []
        people_working = copy.deepcopy(people)
        categories_working = copy.deepcopy(categories)
        if tries == 0:
            output_lines += print_category_selected(categories_working, number_people_wanted)
        output_lines.append("<b>Trial number: {}</b>".format(tries))
        try:
            people_selected, new_output_lines = find_random_sample(categories_working, people_working, columns_data,
                                                                   number_people_wanted, settings.check_same_address,
                                                                   settings.check_same_address_columns,
                                                                   settings.selection_algorithm,
                                                                   test_selection)
            output_lines += new_output_lines
            # check we have reached minimum needed in all cats
            check_min_cat, new_output_lines = check_min_cats(categories_working)
            if check_min_cat:
                output_lines.append("<b>SUCCESS!!</b>")
                success = True
            else:
                output_lines += new_output_lines
        except InfeasibleQuotasError as err:
            output_lines += err.output
            break
        except SelectionError as serr:
            output_lines.append("Failed: Selection Error thrown: " + serr.msg)
        tries += 1
    output_lines.append("Final:")
    output_lines += print_category_selected(categories_working, number_people_wanted)
    if success:
        output_lines.append("We tried {} time(s).".format(tries))
        output_lines.append("Count = {} people selected".format(len(people_selected)))  # , people_selected
    else:
        output_lines.append("Failed {} times... gave up.".format(tries))
    return success, people_selected, output_lines
