import os
import sys
import glob
import re
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    rounds_data = []

    for tr in tables[1].find_all('tr'):
        if 'Number of correct responses' in tr.text or 'Number of wrong responses' in tr.text:
            text = tr.find_all('td')[1].text
            number = re.search(r'\d+', text)
            if number:
                rounds_data.append(int(number.group()))

    return rounds_data

def read_template_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def create_html_table(results, path):
    html_content = """
    <html>
    <head>
    <style>
    table {width: 100%; border-collapse: collapse;}
    td, th {border: 1px solid #ddd; padding: 2px; word-wrap: break-word;}
    td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {max-width: 32px;}
    tr:nth-child(even){background-color: #f2f2f2;}
    tr:hover {background-color: #ddd;}
    th {padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #04AA6D; color: white;}
    pre {white-space: pre-wrap; word-wrap: break-word;}
    </style>
    </head>
    <body>
    <table>
    <tr>
        <th><Q/th><!--Question Number-->
        <th>#&#129351;</th><!--Mean Correct-->
        <th>Question Template</th>
        <th>Solution Template</th>
        <th>Tests Link</th>
    </tr>
    """

    for question, mean_correct in results.items():
        question_template = read_template_file(os.path.join(path, f"Q{question}", "question.txt.template"))
        solution_template = read_template_file(os.path.join(path, f"Q{question}", "solution.py.template"))
        tests_link = os.path.join(path, f"Q{question}", "tests.py.template")

        html_content += f"<tr><td>{question}</td><td>{mean_correct}</td><td><pre>{question_template}</pre></td><td><pre>{solution_template}</pre></td><td><a href='{tests_link}'>Link</a></td></tr>"

    html_content += "</table></body></html>"
    
    with open(os.path.join(path, "all-templates.html"), "w") as file:
        file.write(html_content)

def process_path(path, all_results, all_question_numbers):
    results = {}
    question_numbers = []
    mean_corrects = []

    for i in range(1, 61):
        folder_name = f"Q{i}"
        folder_path = os.path.join(path, folder_name)
        if os.path.exists(folder_path):
            html_file = glob.glob(os.path.join(folder_path, "Final_report_*.html"))
            if html_file:
                with open(html_file[0], 'r') as file:
                    html_content = file.read()
                    data = extract_data(html_content)
                    mean_correct = sum(data[0::2]) / 5
                    results[i] = mean_correct
                    question_numbers.append(i)
                    mean_corrects.append(mean_correct)

    create_html_table(results, path)
    all_results[path] = mean_corrects
    all_question_numbers[path] = question_numbers

def main(paths):
    all_results = {}
    all_question_numbers = {}

    for path in paths:
        process_path(path, all_results, all_question_numbers)

    # Plotting
    plt.figure(figsize=(10, 6))
    for path, mean_corrects in all_results.items():
        question_numbers = all_question_numbers[path]
        plt.plot(question_numbers, mean_corrects, label=f'Mean Correct - {path}')

    plt.xlabel('Question Number')
    plt.ylabel('Mean Correct Responses')
    plt.title('Comparison of Mean Correct Responses Across Different Paths')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path1> <path2> ...")
        sys.exit(1)
    main(sys.argv[1:])
