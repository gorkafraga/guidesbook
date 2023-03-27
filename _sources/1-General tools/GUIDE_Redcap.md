# Redcap - setup 
REDCap is a secure web application for building and managing online surveys and databases. https://www.project-redcap.org/
Ideal for keeping your participants info like screening or questionnaires, which may contain identifying information 

Here are some quick notes: 
## 1. BASIC PROJECT SETUP
### User profile
- Here you can set up your decimal and thousand separator and date formats (will affect data export from all your projects)
-	Each user can have several projects organized in folders. The folder organization of the projects only affects that user 

### Start a new project 
Some important options in project setup are:
-	 Project status. setting up de the project in **“Production”** status means it is considered as completed. It will still be modifiable but to do this would require additional validation steps
-	User **rights**: give permissions and create roles with different levels of permissions. Permit or block data export or designer functions..
-	You can copy, rename project titles, etc
- Adding the participant’s **email**. This will be very helpful to save time contacting participants, sending reminders, etc. Note that some information like email addresses can be **masked** when exporting to prevent identifying subjects. 
-	Backing up project: what is best approach?   (...) See e.g.,  https://guides.temple.edu/c.php?g=936400&p=6879976

## 2.PROJECT DESIGN
### 2.1 INSTRUMENTS, FIELD LABELS AND VARIABLES
Each project can have many different instruments. For example you could have : “cognitive_tests”  and “questionnaires” . 
-	Each **instrument** can be set as survey that can be sent to participants as a link (see next section)
-	Each instrument can have events: for example, several time points of the same test/survey

-	In each instrument you can add **FIELDS** (~= variables)
-	Fields can be added manually  one by one or IMPORTED from an excel sheet. Check redcap ‘data dictionary ‘ to see how attributes and formats will be imported. Importing at least the names and basic formatting might be faster. 
-	You can also upload Fields to and from multiple projects
-	Field label is the field description. It automatically creates a Variable name replacing spaces by “_”. But it is best to revise the variable names and check if that is what you want in your data set (see variable nomenclatures). Decide this names in advance and a consistent way of describing your fields.


- ````WARNING:```` once set you CAN NOT rename your variables or change field labels. It will DELETE all records (participant data) from that variable/field


-	Define the format (date,number , text ,  decimal separator).
-	Add validation to prevent data entry errors  (e.g., percentile scores can only include max 100, date format must comply to the field format, etc)
-	A field can have calculations using a formula to compute variables from existing variables
-	You can add Begin section  to mark a series of fields from , e.g., the same test
-	Set to “required’ if you want to make filling that variable compulsory
-	‘Identifier’ options can be restricted for a given variable to mask exporting (?)
-	Use field Notes for lengthier explanations or notes on this variable .How is this exported?

### 2.2	SURVEY DATA
Surveys are instruments that can be sent via link . 
-	A participant can receive a link with several surveys or you can use individual links
-	The survey filling can be interrupted. If clicked in ‘save and continue later’ the participant will get a code for reentry. If this is lost  the owner can resend this link to the participant
-	Survey distribution tools. If you decide to include the email addresses of your participants (records) this will facilitate sending survey links and reminders to participants. 

### 2.3	CODEBOOK
#### Variable nomenclature 
Keep it consistent and document it !! (use English!)  
E.g.,  Suggested nomenclature for behavioral tests in a longitudinal study:
	[Timepoint_Test_subtest_valueType_scoreType]
              T1_SLRT_words_corr_raw
              T1_SLRT_words_RT_raw

valueType  = if a test has multiple scoring possibilities or errors/hits, RT, accu  
Score type = rw/pr/T …etc

#### Tips 
-	CONSISTENCY across variables: e.g. raw score always indicated the same abbreviation
-	Consistency also with cap/small letters usage! 
-	Keep name parts brief but still descriptive.
-	Write a **Codebook** Variable names should be identifiable on their own, but a codebook is required for completion and clarification clarifications (e.g, ARHQ_father_mean_rw  In codebook ARHQ= Adult reading history questionnaire, etc).
-	Use underscore “_” as separator
-	Avoid spaces and potentially conflicting characters: -,%,&,$,’ etc	

### 2.4	LOGGING CORRECTIONS
Did you find a nonsensical value in one variable and want know who to blame? The tool **Applications/Project Loggin**  will allow this. You can filter by user, time period, etc.

The logging tool may have a lot of info and may not be constantly revised. A suggestion could be to ask people responsible from data entry to log in an easily accessible README text file when and a what corrections have been made to keep track of these (e.g., “10.10.2020 (GFG)– SLRT raw scores from x to y”).   Then this can be confirmed in the detail application logging.
 
