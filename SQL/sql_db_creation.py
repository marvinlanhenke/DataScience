import numpy as np
import sqlalchemy
from faker import Faker
from sqlalchemy import Table, Column, Integer, String, MetaData, Date, ForeignKey

class SQLData:
    def __init__(self, server:str, db:str, uid:str, pwd:str) -> None:
        self.__fake = Faker()
        self.__server = server
        self.__db = db
        self.__uid = uid
        self.__pwd = pwd
        self.__tables = dict()
    
    def connect(self) -> None:
        self.__engine = sqlalchemy.create_engine(
            f"mysql+pymysql://{self.__uid}:{self.__pwd}@{self.__server}/{self.__db}"
        )
        self.__conn = self.__engine.connect()
        self.__meta = MetaData(bind=self.__engine)
    
    def drop_all_tables(self) -> None:
        self.__meta.reflect()
        self.__meta.drop_all(checkfirst=True)
    
    def create_tables(self) -> None:
        self.__tables['jobs'] = Table (
            'jobs', self.__meta,
            Column('job_id', Integer, primary_key=True, autoincrement=True, nullable=False),
            Column('description', String(255))
        )

        self.__tables['companies'] = Table(
            'companies', self.__meta, 
            Column('company_id', Integer, primary_key=True, autoincrement=True, nullable=False),
            Column('name', String(255), nullable=False),
            Column('phrase', String(255)),
            Column('address', String(255)),
            Column('country', String(255)),
            Column('est_date', Date)
        )

        self.__tables['persons'] = Table(
            'persons', self.__meta,
            Column('person_id', Integer, primary_key=True, autoincrement=True, nullable=False),
            Column('job_id', Integer, ForeignKey('jobs.job_id'), nullable=False),
            Column('company_id', Integer, ForeignKey('companies.company_id'), nullable=False),
            Column('last_name', String(255), nullable=False),
            Column('first_name', String(255)),
            Column('date_of_birth', Date),
            Column('address', String(255)),
            Column('country', String(255)),
            Column('zipcode', String(10)),
            Column('salary', Integer)
        )

        self.__meta.create_all()
    
    def populate_tables(self) -> None:
        jobs_ins = list()
        companies_ins = list()
        persons_ins = list()

        for _ in range(100):
            record = dict()
            record['description'] = self.__fake.job()
            jobs_ins.append(record)
        
        for _ in range(100):
            record = dict()
            record['name'] = self.__fake.company()
            record['phrase'] = self.__fake.catch_phrase()
            record['address'] = self.__fake.street_address()
            record['country'] = self.__fake.country()
            record['est_date'] = self.__fake.date_of_birth()
            companies_ins.append(record)

        for _ in range(500):
            record = dict()
            record['job_id'] = np.random.randint(1, 100)
            record['company_id'] = np.random.randint(1, 100)
            record['last_name'] = self.__fake.last_name()
            record['first_name'] = self.__fake.first_name()
            record['date_of_birth'] = self.__fake.date_of_birth()
            record['address'] = self.__fake.street_address()
            record['country'] = self.__fake.country()
            record['zipcode'] = self.__fake.zipcode()
            record['salary'] = np.random.randint(60000, 150000)
            persons_ins.append(record)
        
        self.__conn.execute(self.__tables['jobs'].insert(), jobs_ins)
        self.__conn.execute(self.__tables['companies'].insert(), companies_ins)
        self.__conn.execute(self.__tables['persons'].insert(), persons_ins)


if __name__ == '__main__':
    sql = SQLData('localhost','yourdatabase','root','yourpassword')
    sql.connect()
    sql.drop_all_tables()
    sql.create_tables()
    sql.populate_tables()
