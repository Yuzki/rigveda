import os
import re
import json
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt


class NgramSimilarity:
    def __init__(self, n: int, doc_type="poet"):

        # set directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.text_dir = os.path.join(self.base_dir, "text")

        # settings
        self.n = n
        self.doc_type = doc_type

        # define the range of book or poet
        self.book_list = range(1, 11)
        # self.book_list = list(map(str, range(2, 9)))
        self.poet_list = [
            "Gr̥tsamada",
            "Viśvāmitra Gāthina",
            "Vāmadeva Gautama",
            "Atri Bhauma",
            "Bharadvāja Bārhaspatya",
            "Vasiṣṭha Maitrāvaruṇi",
            "Kaṇva Ghaura",
        ]

        # load text data
        self.text_dict = self._load_text_dict()

    def set_doc_type(self, doc_type):
        """
        Change the document type.

        Args:
            doc_type (str): The new document type. Can be "poet" or "book".
            
        Returns:
            None
        """

        if doc_type in ("poet", "book"):
            self.doc_type = doc_type
        else:
            print("Error: doc_type is 'poet' or 'book'.")

    def _clean_text(self, text):
        """
        Clean the input text by removing certain patterns and extra spaces.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.
        """

        cleaned_text = text
        cleaned_text = re.sub(r"\/!\/|!|\+", "", cleaned_text)
        cleaned_text = re.sub(r"\{.*?\}", "", cleaned_text)
        cleaned_text = cleaned_text.strip()
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        return cleaned_text

    def _load_text_dict(self) -> dict:
        """
        Create text data based on the document type.

        Returns:
            dict: A dictionary containing cleaned text data.
        """

        if self.doc_type == "book":
            book_file_path = os.path.join(self.text_dir, "book.json")
            if os.path.exists(book_file_path):
                with open(book_file_path, "r") as f:
                    dict_text_book = json.load(f)

            else:
                rv_text_path = os.path.join(self.base_dir, "data", "rv_text.csv")
                df = pd.read_csv(rv_text_path, header=0, index_col=0)

                dict_text_book = {}
                for book_num in self.book_list:
                    df_book = df[
                        (df.book == book_num)
                        & (df.transliteration == "tsu")
                        & (df.patha == "padapatha")
                    ]

                    s = " ".join([row.line for row in df_book.itertuples()])
                    cleaned_text = self._clean_text(s)
                    dict_text_book[book_num] = cleaned_text

                with open(book_file_path, mode="w") as f:
                    json.dump(dict_text_book, f, indent=4)

            return dict_text_book

        elif self.doc_type == "poet":
            poet_file_path = os.path.join(self.text_dir, "poet.json")
            if os.path.exists(poet_file_path):
                with open(poet_file_path, mode="r") as f:
                    dict_poet_text = json.load(f)

            else:
                rv_info_path = os.path.join(self.base_dir, "data", "rv_info.csv")
                rv_text_path = os.path.join(self.base_dir, "data", "rv_text.csv")

                dict_poet_text = {}
                df_poet = pd.read_csv(rv_info_path)
                poet_set = set(df_poet["poet"])

                df = pd.read_csv(rv_text_path)

                for poet in poet_set:
                    if poet in self.poet_list:
                        dict_poet_text[poet] = ""
                        for row in df_poet[df_poet["poet"] == poet].itertuples():
                            book = row.bookNum
                            hymn = row.hymnNum
                            verse = row.verseNum

                            for row_p in df[
                                (df["book"] == book)
                                & (df["hymn"] == hymn)
                                & (df["verse"] == verse)
                                & (df["transliteration"] == "tsu")
                                & (df["patha"] == "padapatha")
                            ].itertuples():
                                dict_poet_text[poet] += " " + self._clean_text(
                                    row_p.line
                                )
                dict_poet_text = {
                    key: value.strip() for key, value in dict_poet_text.items()
                }
                with open(poet_file_path, mode="w") as f:
                    json.dump(dict_poet_text, f, indent=4)

            return dict_poet_text

    def _create_ngram_list(self, sentence: str, n: int) -> list:
        """
        Create a list of n-gram text from a given sentence.

        Args:
            sentence (str): The input sentence.
            n (int): The size of the n-grams.

        Returns:
            list: A list of n-gram text.
        """

        word_list = sentence.split(" ")
        ngram_list = [word_list[idx : idx + n] for idx in range(len(word_list) - n + 1)]

        return [" ".join(s) for s in ngram_list]

    def _load_ngram_text_dict(self) -> dict:
        """
        Create n-gram text data (json) and return as a dictionary.

        Returns:
            dict: A dictionary containing n-gram text data.
        """

        file_path = os.path.join(
            self.base_dir, "text", f"ngram_{self.doc_type}_{self.n}.json"
        )
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                ngram_text_dict = json.load(f)
        else:
            ngram_text_dict = {}
            for key, value in self._load_text_dict().items():
                ngram_text_dict[key] = self._create_ngram_list(value, self.n)

            with open(file_path, "w") as f:
                json.dump(ngram_text_dict, f, indent=4)

        self.ngram_text_dict = ngram_text_dict

        print("Load ngram text:", f"{self.n=}", f"{self.doc_type=}")
        return self.ngram_text_dict


    def calculate_similarity(self, ngram1: list, ngram2: list) -> float:
        """
        Calculate the similarity between two sets of n-grams and create the intersection set.

        Args:
            ngram1 (list): A list of label of ngram and n-gram text from the first source.
            ngram2 (list): A list of label of ngram and n-gram text from the second source.

        Returns:
            float: The calculated similarity value.
        """

        label1, ngram_list1 = ngram1
        label2, ngram_list2 = ngram2

        print("Calculate similarity", f"{self.n=}", f"{self.doc_type=}", f"{label1=}", f"{label2=}")

        self.intersection_set = self._get_intersection(ngram_list1, ngram_list2)
        sym_diff_set = set(ngram_list1) ^ set(ngram_list2)

        self.similarity = len(self.intersection_set) / (
            len(self.intersection_set) + len(sym_diff_set)
        )

        self._save_intersection_set(label1, label2)

        return self.similarity
    

    def _save_intersection_set(self, label1, label2):
        file_path = os.path.join(
            self.text_dir,
            "intersection",
            f"intersection_ngram_{self.n}_{label1}-{label2}.txt",
        )
        if not os.path.exists(
            os.path.join(
                self.text_dir, "intersection", f"intersection_ngram_{self.n}_{label2}-{label1}.txt"
            )
        ):
            with open(file_path, "w", encoding="utf-8") as f:
                for text in self.intersection_set:
                    f.write(text + "\n")


    def _get_intersection(self, ngram_list1, ngram_list2) -> set:

        self.intersection_set = set(ngram_list1) & set(ngram_list2)
        
        return self.intersection_set

    def make_heatmap(self):
        """
        Create a heatmap of n-gram similarity and save the figure.

        Returns:
            None
        """

        lst_sim = []

        ngram_text_dict = self._load_ngram_text_dict()
        idx = []
        for i, li in ngram_text_dict.items():
            idx.append(i)
            for j, lj in ngram_text_dict.items():
                sim = self.calculate_similarity([i, li], [j, lj])
                lst_sim.append(sim)

        if self.doc_type == "book":
            ndarray_sim = np.array(lst_sim).reshape(10, 10)
        elif self.doc_type == "poet":
            ndarray_sim = np.array(lst_sim).reshape(7, 7)

        # sim_max = np.amax(ndarray_sim)
        sim_max = np.unique(ndarray_sim)[-2]
        sim_min = np.amin(ndarray_sim)

        # ind = [i for i in range(2, 9)]
        # ind = [i for i in range(1, 11)]
        df = pd.DataFrame(ndarray_sim, index=idx, columns=idx)

        figure_size = 40
        figure_dpi = 200
        plt.figure(figsize=(figure_size, figure_size), dpi=figure_dpi)


        mask = np.zeros_like(df)
        mask[np.triu_indices_from(mask)] = True
        # upp_mat = np.triu(df)
        seaborn.set(font='DejaVu Sans', font_scale=3.0)
        seaborn.heatmap(
            df,
            vmax=sim_max,
            vmin=sim_min,
            square=True,
            cmap="Greys",
            annot=True,
            robust=False,
            mask=mask,
        )

        if self.doc_type == "poet":
            rotation = 45
        elif self.doc_type == "book":
            rotation = 0
        labelsize = 34
        plt.xticks(fontsize=labelsize, rotation=rotation, ha='right')
        plt.yticks(fontsize=labelsize, rotation=rotation, ha='right', va='top')

        fig_path = os.path.join(
            self.base_dir, "fig", f"{self.doc_type}_similarity_ngram_{self.n}.png"
        )
        plt.savefig(fig_path)
        print("Saved heatmap figure:", f"{self.doc_type}")


if __name__ == "__main__":
    for n in range(2, 6):
        ngram = NgramSimilarity(n)

        for doc_type in ("poet", "book"):
            ngram.set_doc_type(doc_type=doc_type)
            ngram.make_heatmap()
