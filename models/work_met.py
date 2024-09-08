import requests
import gc

from bs4 import BeautifulSoup


class Work_met:
    def __init__(self, work_name, images) -> None:
        self.work_name = work_name
        self.images = images

    def request_type(self):
        """Scraps the type of a meteorite based on its deduced work name via the metbull's website.
        Args:
        - work_name -> str : The meteorite's work name

        Returns:
        - mtype : the type of the meteorite as found on the metbull, or None if not found
        """

        self.mtype = None

        headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
            }

        url = f"""
            https://www.lpi.usra.edu/meteor/metbull.php/?sea={self.work_name}&sfor=text&ants=&nwas=\
            &falls=&valids=&stype=contains&lrec=50&map=ge&browse=&country=All&srt=name&categ=All&\
            mblist=All&rect=&phot=&strewn=&snew=0&pnt=Normal%20table&dr=&page=0
        """

        r = requests.get(url=url.strip(), headers=headers)
        r.raise_for_status()  # Raises an HTTPError for bad responses

        soup = BeautifulSoup(r.content, "html.parser")
        main_table = soup.find("table", id="maintable")

        if main_table is not None:
            # Type column location
            header_row = main_table.find("tr")
            headers = header_row.find_all("th")
            type_index = None
            for index, header in enumerate(headers):
                if "Type" in header.text.strip():
                    type_index = index
                    break
            # /Type column location

            # Type
            if type_index is not None:
                for row in main_table.find_all("tr")[1:]:  # Skip the header row
                    cells = row.find_all("td")
                    if len(cells) > type_index:
                        mtype = cells[type_index].text.strip()
                        if "§" in mtype:
                            mtype = mtype.replace("§", "")
                        self.mtype = mtype
            else:
                self.mtype = None
        else:
            self.mtype = None

        gc.collect()

    def __repr__(self) -> str:
        return self.work_name

    def __str__(self) -> str:
        return self.work_name
