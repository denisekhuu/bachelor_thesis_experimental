from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError

class VMUtil(): 
    def __init__(self, configs): 
        """
        :param configs: experiment configurations
        :type configs: Configuration
        """
        self.configs = configs
        self.vm_url = "http://"+ configs.VM_URL
        
    def http_request(self, url, data=None):
        """
        Sends a http request with to an url
        :param url: target url
        :type url: string
        :param data: data for a post request
        :type data: string
        return urllib.Request, object
        """
        try:
            request = Request(url, data=data.encode('ascii', 'ignore')) if data else Request(url)
            response = urlopen(request)
            return request, response
        except HTTPError as e: 
            return None, None
            print("ERROR: {}".format(e))

    def get_data_by(self, name):
        """
        Returns all entries of a metrics 
        :param name: metrics __name__
        :type name: string
        return string
        """
        url = self.vm_url +"/api/v1/query?query=%s{}[2y]"% (name)
        request, response = self.http_request(url)
        return response.read().decode("utf-8")
        
    def push_data(self, data):
        """
        Push data to Victoria Metrics Database
        """
        timestamp = int(datetime.timestamp(datetime.now()))
        url = self.vm_url + "/write?precision=s"
        try:
            request, response = self.http_request(url, data=data.format(timestamp))
        except HTTPError as e: 
            print("ERROR: {}".format(e))