import os
import zipfile
import io
import requests
import luigi

#get top level of project (since this file in located two levels down from top)
PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

class DownloadFile(luigi.Task):
    out_name = luigi.Parameter()
    url = luigi.Parameter()
    filetype = luigi.Parameter(default='str')

    def run(self):
        response = requests.get(self.url, allow_redirects=True)

        if self.filetype == 'str':
            with open(self.output().path, 'wb') as f:
                f.write(response.content)
        elif self.filetype == 'zip':
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(path = self.output().path)
            
    def output(self):
        #tell luigi where to output to
        return luigi.LocalTarget(os.path.join(PROJECT_DIR, self.out_name))

class DownloadAll(luigi.WrapperTask):
    def requires(self):
        yield DownloadFile(url='http://nlp.stanford.edu/data/glove.6B.zip',
                           out_name='data/external/GloveVectors', filetype='zip')
        yield DownloadFile(url='https://onedrive.live.com/download?cid=ABD51044F5341265&resid=ABD51044F5341265%21112053&authkey=AE5A-nrvb2pC2t4',
                           out_name='data/external/jokes.csv', filetype='str')

if __name__ == '__main__':
    luigi.build([DownloadAll()], local_scheduler=True)
