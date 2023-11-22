- aws_tex.py

  1. Create the class and initialize the dynamodb client resources.
  2. Create table name.

  a . create table structure(it's important adding 2 parameters)

  ```
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
  ```

  3. insert data to table:

     a. data_to_insert will be key and value pair(Dict).

     b. For inserting data we gonna mention AttributeType like example here "S" or "N". AttributeType are we mentioned in Table schema.

     ```
     Item={
             "current_date": {"S": data_to_insert["current_date"]},
             "current_time": {"S": data_to_insert["current_time"]},
             "video_index": {"N": str(data_to_insert["video_index"])},
             "cycle_count": {"N": str(data_to_insert["cycle_count"])},
             "cycle_time": {"N": str(data_to_insert["cycle_time"])},
         },
     ```

  4. retrive the data from dynamodb.

