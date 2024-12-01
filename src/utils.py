

def extract_title_and_abstract(text):
    """
    Extracts the title and abstract from a PubMed-style text.

    Parameters:
        text (str): The input text containing the abstract information.

    Returns:
        tuple: A tuple containing the title and the abstract text.
    """
    # Split the text into lines and strip whitespace
    lines = [line.strip() for line in text.split('\n')]

    # Initialize variables
    title = ''
    abstract = ''
    abstract_started = False

    # Find the indices of empty lines
    empty_line_indices = [i for i, line in enumerate(lines) if line == '']

    # Extract the title
    if len(empty_line_indices) >= 2:
        # Title is between the first and second empty lines
        first_empty = empty_line_indices[0]
        second_empty = empty_line_indices[1]
        title_lines = lines[first_empty + 1:second_empty]
        title = ' '.join(title_lines)
    else:
        raise ValueError("The text format is unexpected. Unable to extract the title.")

    # Find the index of "Author information:"
    try:
        author_info_index = lines.index('Author information:')
    except ValueError:
        raise ValueError("Cannot find 'Author information:' in the text.")

    # Find the next empty line after "Author information:"
    for i in range(author_info_index + 1, len(lines)):
        if lines[i] == '':
            abstract_start_index = i + 1
            break
    else:
        # If no empty line is found, start abstract after author info
        abstract_start_index = author_info_index + 1

    # Extract the abstract
    abstract_lines = lines[abstract_start_index:]
    abstract = ' '.join(abstract_lines)

    return title + '\n' + abstract
