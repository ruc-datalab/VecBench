from abc import ABC, abstractmethod
import json

class BaseVD(ABC):
    def __init__(self, config, db_type):
        self.db_type = db_type
        self.config = config[db_type]
        self.connection = None

    @abstractmethod
    def connect(self):
        """Establish a connection to the database."""
        pass

    @abstractmethod
    def create_table(self, table_name, schema):
        """Create a database table or collection."""
        pass

    @abstractmethod
    def create_index(self, index_params, args):
        """Create a vector index."""
        pass

    @abstractmethod
    def create_scalar_index(self, index_params, args):
        """Create a scalar index."""
        pass

    @abstractmethod
    def process_data(self, data, table_name, schema):
        """Preprocess or transform data before insertion."""
        pass

    @abstractmethod
    def insert_data(self, data, db, schema):
        """Insert data into the database."""
        pass

    @abstractmethod
    def query(self, query):
        """Execute a query on the database."""
        pass

    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass

    def load_ground_truth(self, ground_truth_file):
        """Load ground truth data from a JSON file."""
        with open(ground_truth_file, 'r') as f:
            return json.load(f)

    def close_connection(self):
        """Close an active database connection if it exists."""
        if self.connection:
            self.connection.close()
