"""
Script of the skillsChecker class 
author: Valentin Barriere
12/22
"""
import nltk 
import re 
from tqdm import tqdm 
import random 

# POS-tagging
import spacy 
nlp = spacy.load("en_core_web_sm")

try: 
    from nltk import word_tokenize
except:
    nltk.download('punkt')
    from nltk import word_tokenize
tokenizer = lambda x: word_tokenize(x)

def remove_elt(l, e):
    l.remove(e)
    return l

class skillsChecker():
    """
    Class to detect if a skill is present in a text, and change it accoordingly. 
    
    This version is not using regex to detect and change words, it should work for gender, color, counting. 
    It will not be efficient for the emotion. To test what is present in the caption dataset. 
    
    ```
    self.list_skills = {
        skill_i : [token_i_1, token_i_2, ...], 
        skill_j : list_tokens_to_check_j, 
        ...
    }
    ```
    
    ```
    self.list_change = {
        skill_i : { 
            token_i_1 : list_possible_change_for_token_i_1,
            token_i_2 : list_possible_change_for_token_i_2,
            ...
        },
        skill_j  : { 
            token_j_1 : list_possible_change_for_token_j_1,
            ...
        },
        ...
    }
    ```
    
    """
    def __init__(self):
        
        self.male = ['boy', 'boys', 'man', 'men', 'guy', 'guys']
        self.female = ['girl', 'girls', 'woman', 'women']
        
        # list of words to check
        self.list_skills = {
            'gender' : self.male + self.female, 
            'emotion' : ['angry', 'happy', 'sad'],  # easier with visual emotion recognition ? # without disgust, fear and surprise
            'counting' : ['two', 'three', 'four', 'five', 'six'],
            'color' : ['red', 'green', 'blue', 'yellow', 'purple'] #+ ['black', 'white', 'orange']
            ,
        }
        self.list_change = {
            'gender': {
                'boy' : ['girl'], 
                'boys' : ['girls'], 
                'man' : ['woman'], 
                'men' : ['women'], 
                'guy' : ['girl'], 
                'guys' : ['girls'],
                'girl' : ['boy'], 
                'girls' : ['boys'], 
                'woman' : ['man'], 
                'women' : ['men'],
            },
            'color' : { # hopefully no MWE like green light, hard to detect with pos-tagger... 
                color : remove_elt(self.list_skills['color'].copy(), color) for color in self.list_skills['color']
            },
            'counting' : {
                'two' : ['three'],
                'three' : ['two', 'four'],
                'four' : ['three', 'five'],
                'five' : ['four', 'six'],
                'six' : ['five']
            },
            'emotion' : { # hard to see how well this will work... to test! 
                emotion : remove_elt(self.list_skills['emotion'].copy(), emotion) for emotion in self.list_skills['emotion']
            }
        }
    
    def find_captions_skill(self, list_captions, skill, verbose=False):
        """
        find the captions that contains a special skill
        output:
        list_captions_kept: list of the captions containing the skill
        list_words_skill: list of boolean lists regarding which skill-related word was found in each sentence   
        """
        if verbose: list_captions = tqdm(list_captions)
            
        # list of boolean values regarding the skill is here
        list_bool_skill = []
        list_words_skill = []
        
        for capt in list_captions:
            # tokenize and put in lower in order to find the words
            capt = [tok.lower() for tok in tokenizer(capt)]
            
            list_booleans = [skill_wd in capt for skill_wd in self.list_skills[skill]]
            is_skill = any(list_booleans)
            list_bool_skill.append(is_skill)
            if is_skill:
                # list_words_skill.append(self.list_skills[skill][list_booleans])
                list_words_skill.append([a for a, b in zip(self.list_skills[skill], list_booleans) if b])
            
        list_captions_kept = [a for a, b in zip(list_captions, list_bool_skill) if b]
        
        return list_captions_kept, list_words_skill
    
    def find_captions_all_skills(self, list_captions):
        """
        All skills at once 
        """
        dict_bool_skills = {sk : [] for sk in self.list_skills.keys()}
        dict_words_skills = {sk : [] for sk in self.list_skills.keys()}

        for sk, skill_wds in list_skills.items():
            dict_captions_skills[sk], dict_words_skills[sk] = self.find_captions_skill(list_captions, sk)
                
        return dict_captions_skills, dict_words_skills
    
    def change_pronoun_gender(self, new_caption, wd):
        """
        Change his/him/her
        new_caption is the caption already changed
        wd is the word that has been changed
        """
        if re.findall(r'\bhis\b|\bher\b|\bhim\b', new_caption):
            # if female to male
            if wd in self.female:
                # ('him', 'PRP') and ('his', 'PRP$')
                tag_her = [wd.tag_ for wd in nlp(new_caption) if wd.text == 'her']
                # sometimes there is a woman but it's a 'him' or a 'his' that is detected
                if len(tag_her):
                    if tag_her[0] == 'PRP$':
                        pro_to_sub = 'his' 
                    else:
                        pro_to_sub = 'him'
                    new_caption = re.sub(r'\b%s\b'%'her', pro_to_sub, new_caption)
                # if male to female
            else:
                new_caption = re.sub(r'\b%s\b'%'him', 'her', new_caption)
                new_caption = re.sub(r'\b%s\b'%'his', 'her', new_caption)

        return new_caption

    def change_captions_skill(self, list_captions, skill, verbose=False):
        """
        output: 
        list_captions_skill_changed: list of n-uples containing the new captions for the skill in question 
        """
        
        list_captions_skill, list_words_skill = self.find_captions_skill(list_captions, skill, verbose=verbose)
        
        # all the captions changed
        list_captions_skill_changed = []
        
        zip_it = tqdm(zip(list_captions_skill, list_words_skill)) if verbose else zip(list_captions_skill, list_words_skill)
        
        for capt, list_word_skill in zip_it:
            
            # one caption many variations 
            list_one_caption_changed = []
            for wd in list_word_skill:  
                wd_to_sub = random.choice(self.list_change[skill][wd])
                
                # to keep the capitalization of the first letter
                # new_caption = re.sub(r'\b[%s%s]%s\b'%(wd[0].upper(), wd[0], wd[1:]), wd_to_sub, capt)
                new_caption = re.sub(r'\b%s\b'%wd, wd_to_sub, capt)
                if new_caption == capt:
                    wd = wd.capitalize()
                    wd_to_sub = wd_to_sub.capitalize()
                    new_caption = re.sub(r'\b%s\b'%wd, wd_to_sub, capt)
                    
                # pronoun for gender 
                if skill == 'gender': new_caption = self.change_pronoun_gender(new_caption, wd)
                    
                list_one_caption_changed.append(new_caption)
                # print(wd, wd_to_sub, capt, '\n', list_one_caption_changed[-1])
            list_captions_skill_changed.append(list_one_caption_changed)
            
        return [(a,b) for a,b in zip(list_captions_skill, list_captions_skill_changed)], list_words_skill
    
if __name__ =='__main__':
    
    list_captions = ['Two guys in the beach next to a firecamp', 'Three women in a swimming pool', 
                 'The saddiest movie ever', 'A very happy boy', 'An asadandhappyo on a red carpet', 
                 'Beautiful red and yellow sky']
    
    checker = skillsChecker()
    
    print(checker.change_captions_skill(list_captions, 'gender'), '\n',
          checker.change_captions_skill(list_captions, 'color'), '\n',          
          checker.change_captions_skill(list_captions, 'counting'))