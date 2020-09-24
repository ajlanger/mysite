from django.shortcuts import render, redirect
from .models import processText
from django.http import HttpResponse, HttpResponseRedirect
from .official_extractor.decision_logic_extractor_functions import decision_logic_extractor

def flatten(nested_list):
    flat_list = []
    if is_empty(nested_list):
        return nested_list
    if type(nested_list) != list and type(nested_list) != tuple:
        try:
            return nested_list.text
        except:
            return nested_list

    for sublist in nested_list:
        if type(sublist) == list or type(sublist) == tuple:
            for item in sublist:
                try:
                    flat_list.append(item.text)
                except:
                    flat_list.append(item)
        else:
            try:
                flat_list.append(sublist.text)
            except:
                flat_list.append(sublist)
    for el in flat_list:
        if type(el) == list or type(sublist) == tuple:
            return flatten(flat_list)
    return flat_list


def is_empty(lst_tup):
    if lst_tup:
        return False
    else:
        return True


def make_string(inputlist):
    flatlist = flatten(inputlist)
    if is_empty(flatlist):
        return flatlist
    return ' '.join(flatlist)


# Create your views here.
def dle_view(request, *args, **kwargs):
    input_text   = ""
    detail_level = ""

    input_objects = processText(request.POST or None)

    if input_objects.is_valid():
        input_text   = input_objects.cleaned_data.get("input_text")
        detail_level = input_objects.cleaned_data.get("detail_level")

        print(input_text, detail_level)

        answer    = decision_logic_extractor(input_text)

        high_info = []
        low_info  = []

        original_sentences = [s[f'sentence {i+1}'] for i, s in enumerate(answer)]

        for s in answer:
            if s['high'] != 'None' and s['high'] and type(s['high']) == dict:
                high_info.extend([{'condition': make_string(s['high']['condition']),
                                   'consequence': make_string(s['high']['consequence'])}])
            elif s['high'] == 'No conditional statements could be extracted in spite of a condition being present.':
                high_info.extend([{'condition': 'Error during extraction',
                                   'consequence': 'Error during extraction'}])
            else:
               high_info.extend([{'condition':'None', 'consequence': 'None'}])

            if s['low'] != 'None' and type(s['high']) == dict:
                low_info.extend([{'if': make_string(s['low']['if']),
                                  'then': make_string(s['low']['then']),
                                  'else': make_string(s['low']['else'])}])
            elif s['low'] == 'error during extraction':
                low_info.extend([{'if': 'Error during extraction',
                                  'then': 'Error during extraction',
                                  'else': 'Error during extraction'}])
            else:
                low_info.extend([{'if':'None', 'then': 'None', 'else': 'None'}])

        sentence_ids     = [i+1 for i in range(len(original_sentences))]
        conditions      = [s['condition'] for s in high_info]
        consequences    = [s['consequence'] for s in high_info]
        if_statements   = [s['if'] for s in low_info]
        then_statements = [s['then'] for s in low_info]
        else_statements = [s['else'] if s['else'] else 'None' for s in low_info]


        sentences_info = zip(sentence_ids, original_sentences, conditions, consequences, if_statements, then_statements, else_statements)

        # input_text   = ""
        # detail_level = ""
        #
        input_objects = processText(request.POST or None)

        context = {'display_outputs': True,
                   'detail_level': detail_level,
                   'input_objects': input_objects,
                   'sentences_info': sentences_info}

        # Process the text and give output
        return render(request, 'dl_extractor.html', context)
    else:
        context = {'display_outputs': False,
                   'detail_level': None,
                   'input_objects': input_objects,
                   'sentences_info': []}

        return render(request, 'dl_extractor.html', context)
