import boto3
import pandas as pd

class Bucketdynamodb:
    def __init__(self, table_name):
        self.s3_resource = boto3.client("dynamodb")
        self.table_name = table_name

    def create_table(self):
        self.existing_tables = self.s3_resource.list_tables()["TableNames"]
        # print(self.existing_tables)

        if self.table_name not in self.existing_tables:
            # Define the table schema and provisioned throughput
            table_definition = {
                "TableName": self.table_name,
                "KeySchema": [
                    {
                        "AttributeName": "current_date",
                        "KeyType": "HASH",  # Assuming 'S' for string
                    },
                    {
                        "AttributeName": "current_time",
                        "KeyType": "RANGE",  # Assuming 'S' for string
                    },
                ],
                "AttributeDefinitions": [
                    {
                        "AttributeName": "current_date",  # Partition Key
                        "AttributeType": "S",
                    },
                    {"AttributeName": "current_time", "AttributeType": "S"},
                ],
                "ProvisionedThroughput": {
                    "ReadCapacityUnits": 5,
                    "WriteCapacityUnits": 5,
                },
            }
            # Create the table
            response = self.s3_resource.create_table(**table_definition)
            # print(response)
            print("table created")
        else:
            print("table already exists")

    def insert_data_to_table(self, data_to_insert):
        # Sample data to be inserted

        # Insert data into the table
        response = self.s3_resource.put_item(
            TableName=self.table_name,
            Item={
                "current_date": {"S": data_to_insert["current_date"]},
                "current_time": {"S": data_to_insert["current_time"]},
                "video_index": {"N": str(data_to_insert["video_index"])},
                "cycle_count": {"N": str(data_to_insert["cycle_count"])},
                "cycle_time": {"N": str(data_to_insert["cycle_time"])},
            },
        )
        print(response)

    def retrive_the_data(self):

        # Use the get_item method to retrieve the item
        response = self.s3_resource.scan(TableName=self.table_name)

        # Print the retrieved item
        tex_total_data = []
        for item in response["Items"]:
            tex_total_data.append(item)
            # print(item)
        data = pd.DataFrame(tex_total_data)
        # print(data)