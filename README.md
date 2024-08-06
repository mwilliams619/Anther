# Anther Project

## Project Overview

The Anther project aims to assist artists in evaluating the relationship between their songs and other content available on streaming platforms. The primary goal is to develop a service that enables artists to efficiently assess the compatibility of their songs with playlists.

## Key Features

- Search functionality to retrieve songs and playlists from the database
- Song processing to extract relevant data for analysis
- Comparison algorithm to cluster songs based on similarity
- Interactive D3 dashboard to visualize song clusters

## Technical Stack

- **Backend**: Python, Django
- **Frontend**: jQuery, HTML, CSS
- **Database**: PostgreSQL
- **Visualization**: D3.js

## Setup Local Development Environment

### Prerequisites

- PostgreSQL
- Postico2
- Python (version X.X or higher)

### Steps

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/anther-project.git
   cd anther-project
   ```

2. Set up the local settings:

   - Copy the content of `localsettingstemplate.py` into a new file called `localsettings.py`
   - Update `localsettings.py` with your local PostgreSQL environment details

3. Update the `launch.json` file with your local `manage.py` path

4. Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

5. Install required packages:

   ```
   pip install -r requirements.txt
   ```

6. Set up the database:

   ```
   python manage.py makemigrations
   python manage.py migrate
   ```

## Usage

The repo should run as is as a sandbox. To add new playlists for comparisons, navigate to the /search page in your browser and search new playlists. This should populate songs from the playlist and allow you to view them in the /playlists page.

## Contributing

To contribute to Anther, clone this repo locally and please commit your code on a seperate branch.

If you're making core library changes please write unit tests for your  code, and check that everything works  before  opening a pull-request

## License

```
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

If you need to get in touch please contact me on [Linkedin](https://www.linkedin.com/in/mattwilliams19/)

---

Feel free to star ⭐ this repository if you find it useful!

