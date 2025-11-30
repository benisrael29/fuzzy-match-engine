"""
Comprehensive English nickname mapping dictionary.

Maps common nicknames to their canonical full names for name normalization.
All keys are lowercase for case-insensitive matching.
"""

NICKNAME_MAP = {
    # William variations
    'will': 'william',
    'bill': 'william',
    'billy': 'william',
    'willie': 'william',
    'willy': 'william',
    'wil': 'william',
    
    # Robert variations
    'bob': 'robert',
    'bobby': 'robert',
    'rob': 'robert',
    'robbie': 'robert',
    'robby': 'robert',
    'bert': 'robert',
    'bobbie': 'robert',
    
    # David variations
    'dave': 'david',
    'davy': 'david',
    'davey': 'david',
    
    # Michael variations
    'mike': 'michael',
    'mikey': 'michael',
    'mick': 'michael',
    'mickey': 'michael',
    'micky': 'michael',
    
    # James variations
    'jim': 'james',
    'jimmy': 'james',
    'jamie': 'james',
    'jimmie': 'james',
    
    # Joseph variations
    'joe': 'joseph',
    'joey': 'joseph',
    'jos': 'joseph',
    
    # Thomas variations
    'tom': 'thomas',
    'tommy': 'thomas',
    'thom': 'thomas',
    
    # Richard variations
    'dick': 'richard',
    'rick': 'richard',
    'ricky': 'richard',
    'rich': 'richard',
    'richie': 'richard',
    'rickie': 'richard',
    
    # John variations
    'jack': 'john',
    'johnny': 'john',
    'jon': 'john',
    'jonny': 'john',
    'johny': 'john',
    
    # Christopher variations
    'chris': 'christopher',
    'christo': 'christopher',
    'kit': 'christopher',
    
    # Daniel variations
    'dan': 'daniel',
    'danny': 'daniel',
    'dannie': 'daniel',
    
    # Edward variations
    'ed': 'edward',
    'eddie': 'edward',
    'eddy': 'edward',
    'ted': 'edward',
    'teddy': 'edward',
    'ned': 'edward',
    
    # Francis variations
    'frank': 'francis',
    'frankie': 'francis',
    'fran': 'francis',
    
    # Henry variations
    'harry': 'henry',
    'hank': 'henry',
    
    # Kenneth variations
    'ken': 'kenneth',
    'kenny': 'kenneth',
    'kennie': 'kenneth',
    
    # Lawrence variations
    'larry': 'lawrence',
    'lawrence': 'lawrence',
    'lorry': 'lawrence',
    
    # Matthew variations
    'matt': 'matthew',
    'matty': 'matthew',
    'mat': 'matthew',
    
    # Nicholas variations
    'nick': 'nicholas',
    'nickie': 'nicholas',
    'nicki': 'nicholas',
    'nico': 'nicholas',
    
    # Patrick variations
    'pat': 'patrick',
    'paddy': 'patrick',
    'patsy': 'patrick',
    
    # Ronald variations
    'ron': 'ronald',
    'ronnie': 'ronald',
    'ronny': 'ronald',
    
    # Samuel variations
    'sam': 'samuel',
    'sammy': 'samuel',
    'sammie': 'samuel',
    
    # Steven/Stephen variations
    'steve': 'steven',
    'steven': 'steven',
    'stephen': 'steven',
    'stevie': 'steven',
    
    # Anthony variations
    'tony': 'anthony',
    'anthony': 'anthony',
    'ant': 'anthony',
    
    # Albert variations
    'al': 'albert',
    'albert': 'albert',
    'bert': 'albert',
    
    # Alexander variations
    'alex': 'alexander',
    'alexander': 'alexander',
    'alexandr': 'alexander',
    'sandy': 'alexander',
    
    # Andrew variations
    'andy': 'andrew',
    'andrew': 'andrew',
    'drew': 'andrew',
    
    # Benjamin variations
    'ben': 'benjamin',
    'benny': 'benjamin',
    'benjie': 'benjamin',
    
    # Bradley variations
    'brad': 'bradley',
    'bradley': 'bradley',
    
    # Charles variations
    'charlie': 'charles',
    'chuck': 'charles',
    'charley': 'charles',
    'chas': 'charles',
    
    # Donald variations
    'don': 'donald',
    'donald': 'donald',
    'donny': 'donald',
    'donnie': 'donald',
    
    # Douglas variations
    'doug': 'douglas',
    'douglas': 'douglas',
    'dougie': 'douglas',
    
    # Frederick variations
    'fred': 'frederick',
    'freddie': 'frederick',
    'freddy': 'frederick',
    'fritz': 'frederick',
    
    # Gregory variations
    'greg': 'gregory',
    'gregory': 'gregory',
    'gregor': 'gregory',
    
    # Jeffrey variations
    'jeff': 'jeffrey',
    'jeffrey': 'jeffrey',
    'jeffery': 'jeffrey',
    'geoff': 'jeffrey',
    'geoffrey': 'jeffrey',
    
    # Gerald variations
    'jerry': 'gerald',
    'gerry': 'gerald',
    'gerard': 'gerald',
    
    # Joshua variations
    'josh': 'joshua',
    'joshua': 'joshua',
    'joshie': 'joshua',
    
    # Mark variations
    'mark': 'mark',
    'marc': 'mark',
    'marcus': 'mark',
    
    # Paul variations
    'paul': 'paul',
    'pauly': 'paul',
    
    # Peter variations
    'pete': 'peter',
    'peter': 'peter',
    'petey': 'peter',
    
    # Philip variations
    'phil': 'philip',
    'philip': 'philip',
    'phillip': 'philip',
    
    # Raymond variations
    'ray': 'raymond',
    'raymond': 'raymond',
    'raymon': 'raymond',
    
    # Scott variations
    'scott': 'scott',
    'scotty': 'scott',
    'scot': 'scott',
    
    # Sean variations
    'sean': 'sean',
    'shawn': 'sean',
    'shane': 'sean',
    'shaun': 'sean',
    
    # Timothy variations
    'tim': 'timothy',
    'timmy': 'timothy',
    'timmie': 'timothy',
    
    # Todd variations
    'todd': 'todd',
    'toddy': 'todd',
    
    # Elizabeth variations
    'beth': 'elizabeth',
    'liz': 'elizabeth',
    'lizzie': 'elizabeth',
    'lizzy': 'elizabeth',
    'betsy': 'elizabeth',
    'betty': 'elizabeth',
    'bess': 'elizabeth',
    'bessie': 'elizabeth',
    'lisa': 'elizabeth',
    'liza': 'elizabeth',
    'eliza': 'elizabeth',
    'elsie': 'elizabeth',
    'ella': 'elizabeth',
    
    # Katherine/Catherine variations
    'kate': 'katherine',
    'katie': 'katherine',
    'katy': 'katherine',
    'cathy': 'catherine',
    'cath': 'catherine',
    'kath': 'katherine',
    'kathy': 'katherine',
    'kathie': 'katherine',
    'katharine': 'katherine',
    'catharine': 'catherine',
    'kay': 'katherine',
    'kitty': 'katherine',
    
    # Ann/Anne/Anna variations
    'ann': 'ann',
    'anne': 'anne',
    'anna': 'anna',
    'annie': 'ann',
    'nan': 'ann',
    'nancy': 'nancy',
    
    # Barbara variations
    'barb': 'barbara',
    'barbie': 'barbara',
    'barbara': 'barbara',
    'babs': 'barbara',
    
    # Carol/Caroline/Carolyn variations
    'carol': 'carol',
    'caroline': 'caroline',
    'carolyn': 'carolyn',
    'carrie': 'carol',
    'cassie': 'carolyn',
    
    # Deborah variations
    'deb': 'deborah',
    'debbie': 'deborah',
    'debra': 'deborah',
    'debby': 'deborah',
    'debi': 'deborah',
    
    # Jennifer variations
    'jen': 'jennifer',
    'jenny': 'jennifer',
    'jenn': 'jennifer',
    'jennie': 'jennifer',
    'jennifer': 'jennifer',
    
    # Jessica variations
    'jess': 'jessica',
    'jessie': 'jessica',
    'jessica': 'jessica',
    'jessy': 'jessica',
    
    # Margaret variations
    'marg': 'margaret',
    'maggie': 'margaret',
    'meg': 'margaret',
    'peg': 'margaret',
    'peggy': 'margaret',
    'marge': 'margaret',
    'margie': 'margaret',
    'margy': 'margaret',
    'daisy': 'margaret',
    'madge': 'margaret',
    
    # Patricia variations
    'pat': 'patricia',
    'patty': 'patricia',
    'patti': 'patricia',
    'patsy': 'patricia',
    'trish': 'patricia',
    'tricia': 'patricia',
    'patricia': 'patricia',
    
    # Susan variations
    'sue': 'susan',
    'susie': 'susan',
    'suzie': 'susan',
    'susan': 'susan',
    'suzanne': 'susan',
    'suzan': 'susan',
    
    # Mary variations
    'mary': 'mary',
    'marie': 'marie',
    'maria': 'maria',
    'molly': 'mary',
    'polly': 'mary',
    'mae': 'mary',
    'may': 'mary',
    'mamie': 'mary',
    
    # Linda variations
    'linda': 'linda',
    'lynn': 'linda',
    'lynda': 'linda',
    
    # Additional common names
    'alice': 'alice',
    'allie': 'alice',
    'ally': 'alice',
    
    'amy': 'amy',
    'aimie': 'amy',
    
    'angela': 'angela',
    'angie': 'angela',
    'angel': 'angela',
    
    'ashley': 'ashley',
    'ash': 'ashley',
    
    'brenda': 'brenda',
    'brendie': 'brenda',
    
    'christina': 'christina',
    'christine': 'christine',
    'chris': 'christina',
    'tina': 'christina',
    'chrissy': 'christina',
    
    'cynthia': 'cynthia',
    'cindy': 'cynthia',
    'cyndi': 'cynthia',
    
    'diana': 'diana',
    'diane': 'diane',
    'dianne': 'diane',
    'di': 'diana',
    
    'donna': 'donna',
    'donnie': 'donna',
    
    'dorothy': 'dorothy',
    'dot': 'dorothy',
    'dottie': 'dorothy',
    'dotty': 'dorothy',
    
    'emily': 'emily',
    'em': 'emily',
    'emmie': 'emily',
    'emma': 'emily',
    
    'frances': 'frances',
    'fran': 'frances',
    'frankie': 'frances',
    
    'helen': 'helen',
    'helena': 'helen',
    'nell': 'helen',
    'nellie': 'helen',
    
    'janet': 'janet',
    'jan': 'janet',
    'jane': 'jane',
    'janice': 'janice',
    'janis': 'janice',
    
    'joan': 'joan',
    'joanne': 'joanne',
    'joanna': 'joanna',
    
    'judith': 'judith',
    'judy': 'judith',
    'judie': 'judith',
    
    'julia': 'julia',
    'julie': 'julia',
    'juliet': 'julia',
    
    'karen': 'karen',
    'karin': 'karen',
    'caren': 'karen',
    
    'kathleen': 'kathleen',
    'kathy': 'kathleen',
    'kathie': 'kathleen',
    'katy': 'kathleen',
    
    'kimberly': 'kimberly',
    'kim': 'kimberly',
    'kimmie': 'kimberly',
    
    'laura': 'laura',
    'laurie': 'laura',
    'lauren': 'laura',
    
    'lisa': 'lisa',
    'liz': 'lisa',
    'liza': 'lisa',
    
    'lori': 'lori',
    'lorie': 'lori',
    'lory': 'lori',
    
    'michelle': 'michelle',
    'shell': 'michelle',
    'shelly': 'michelle',
    
    'nicole': 'nicole',
    'niki': 'nicole',
    'nikki': 'nicole',
    'nic': 'nicole',
    
    'pamela': 'pamela',
    'pam': 'pamela',
    'pammie': 'pamela',
    
    'rebecca': 'rebecca',
    'becky': 'rebecca',
    'becca': 'rebecca',
    'beckie': 'rebecca',
    
    'sandra': 'sandra',
    'sandy': 'sandra',
    'sandi': 'sandra',
    
    'sarah': 'sarah',
    'sara': 'sarah',
    'sally': 'sarah',
    'sal': 'sarah',
    
    'sharon': 'sharon',
    'sharron': 'sharon',
    'sharyn': 'sharon',
    
    'stephanie': 'stephanie',
    'steph': 'stephanie',
    'stefanie': 'stephanie',
    
    'teresa': 'teresa',
    'terry': 'teresa',
    'tess': 'teresa',
    'tessa': 'teresa',
    
    'virginia': 'virginia',
    'ginny': 'virginia',
    'virgie': 'virginia',
    
    # Additional male names
    'adrian': 'adrian',
    'adriano': 'adrian',
    
    'alan': 'alan',
    'allen': 'alan',
    'allan': 'alan',
    
    'arthur': 'arthur',
    'art': 'arthur',
    'artie': 'arthur',
    
    'brian': 'brian',
    'bryan': 'brian',
    'bryant': 'brian',
    
    'bruce': 'bruce',
    
    'caleb': 'caleb',
    'cal': 'caleb',
    
    'carl': 'carl',
    'carlos': 'carl',
    'charles': 'charles',
    
    'christian': 'christian',
    'chris': 'christian',
    
    'cody': 'cody',
    'codey': 'cody',
    
    'connor': 'connor',
    'conor': 'connor',
    
    'dylan': 'dylan',
    'dillon': 'dylan',
    
    'eric': 'eric',
    'erik': 'eric',
    'erick': 'eric',
    
    'evan': 'evan',
    'evan': 'evan',
    
    'gabriel': 'gabriel',
    'gabe': 'gabriel',
    'gabby': 'gabriel',
    
    'george': 'george',
    'georgie': 'george',
    
    'ian': 'ian',
    'iain': 'ian',
    
    'isaac': 'isaac',
    'ike': 'isaac',
    
    'jacob': 'jacob',
    'jake': 'jacob',
    'jakey': 'jacob',
    
    'jason': 'jason',
    'jase': 'jason',
    
    'jonathan': 'jonathan',
    'jon': 'jonathan',
    'johnny': 'jonathan',
    
    'justin': 'justin',
    'just': 'justin',
    
    'kyle': 'kyle',
    'kile': 'kyle',
    
    'lucas': 'lucas',
    'luke': 'lucas',
    
    'mason': 'mason',
    'mase': 'mason',
    
    'nathan': 'nathan',
    'nate': 'nathan',
    'nat': 'nathan',
    
    'noah': 'noah',
    
    'oliver': 'oliver',
    'ollie': 'oliver',
    'olie': 'oliver',
    
    'oscar': 'oscar',
    'oskar': 'oscar',
    
    'owen': 'owen',
    
    'ryan': 'ryan',
    'ryann': 'ryan',
    
    'tyler': 'tyler',
    'ty': 'tyler',
    
    'victor': 'victor',
    'vic': 'victor',
    'vick': 'victor',
    
    'william': 'william',
    'zachary': 'zachary',
    'zach': 'zachary',
    'zac': 'zachary',
    'zack': 'zachary',
    'zackary': 'zachary',
    'zackery': 'zachary',
    
    # Additional common male names and variations
    'aaron': 'aaron',
    'aron': 'aaron',
    'ronnie': 'aaron',
    
    'adam': 'adam',
    'addy': 'adam',
    
    'adrian': 'adrian',
    'adriano': 'adrian',
    'adrien': 'adrian',
    
    'austin': 'austin',
    'austen': 'austin',
    
    'brandon': 'brandon',
    'branden': 'brandon',
    'bran': 'brandon',
    
    'brendan': 'brendan',
    'brendon': 'brendan',
    'bren': 'brendan',
    
    'brent': 'brent',
    'bren': 'brent',
    
    'caleb': 'caleb',
    'cal': 'caleb',
    'kaleb': 'caleb',
    
    'cameron': 'cameron',
    'cam': 'cameron',
    'cammy': 'cameron',
    
    'carlos': 'carlos',
    'carl': 'carlos',
    'charlie': 'carlos',
    
    'carter': 'carter',
    'cart': 'carter',
    
    'chase': 'chase',
    'chas': 'chase',
    
    'christian': 'christian',
    'chris': 'christian',
    'christo': 'christian',
    
    'colin': 'colin',
    'collin': 'colin',
    'col': 'colin',
    
    'connor': 'connor',
    'conor': 'connor',
    'con': 'connor',
    
    'cory': 'cory',
    'corey': 'cory',
    'corie': 'cory',
    
    'damian': 'damian',
    'damien': 'damian',
    'damon': 'damian',
    
    'derek': 'derek',
    'derrick': 'derek',
    'rick': 'derek',
    
    'devin': 'devin',
    'devon': 'devin',
    'dev': 'devin',
    
    'dillon': 'dillon',
    'dylan': 'dillon',
    'dill': 'dillon',
    
    'dominic': 'dominic',
    'dom': 'dominic',
    'dominic': 'dominic',
    'nick': 'dominic',
    
    'eli': 'eli',
    'elias': 'eli',
    'elijah': 'eli',
    
    'ethan': 'ethan',
    'eth': 'ethan',
    
    'evan': 'evan',
    'ev': 'evan',
    
    'gavin': 'gavin',
    'gav': 'gavin',
    
    'graham': 'graham',
    'gram': 'graham',
    
    'grant': 'grant',
    'gran': 'grant',
    
    'harrison': 'harrison',
    'harry': 'harrison',
    'harris': 'harrison',
    
    'hunter': 'hunter',
    'hunt': 'hunter',
    
    'isaac': 'isaac',
    'ike': 'isaac',
    'izzy': 'isaac',
    
    'ivan': 'ivan',
    'vanya': 'ivan',
    
    'jackson': 'jackson',
    'jack': 'jackson',
    'jax': 'jackson',
    'jaxson': 'jackson',
    
    'jaden': 'jaden',
    'jade': 'jaden',
    'jayden': 'jaden',
    
    'jameson': 'jameson',
    'jamie': 'jameson',
    'jim': 'jameson',
    
    'jordan': 'jordan',
    'jordy': 'jordan',
    'jordie': 'jordan',
    
    'julian': 'julian',
    'jules': 'julian',
    'julio': 'julian',
    
    'kevin': 'kevin',
    'kev': 'kevin',
    'kevvy': 'kevin',
    
    'landon': 'landon',
    'landy': 'landon',
    
    'levi': 'levi',
    'lev': 'levi',
    
    'logan': 'logan',
    'log': 'logan',
    
    'louis': 'louis',
    'lou': 'louis',
    'lewis': 'louis',
    
    'lucas': 'lucas',
    'luke': 'lucas',
    'lukas': 'lucas',
    
    'marcus': 'marcus',
    'mark': 'marcus',
    'marco': 'marcus',
    
    'max': 'max',
    'maxwell': 'max',
    'maximilian': 'max',
    'maximus': 'max',
    
    'miles': 'miles',
    'myles': 'miles',
    
    'nathaniel': 'nathaniel',
    'nate': 'nathaniel',
    'nat': 'nathaniel',
    'nathan': 'nathaniel',
    
    'nicholas': 'nicholas',
    'nick': 'nicholas',
    'nico': 'nicholas',
    'colin': 'nicholas',
    
    'noah': 'noah',
    'noe': 'noah',
    
    'oscar': 'oscar',
    'oskar': 'oscar',
    'ozzy': 'oscar',
    
    'owen': 'owen',
    'o': 'owen',
    
    'parker': 'parker',
    'park': 'parker',
    
    'patrick': 'patrick',
    'pat': 'patrick',
    'paddy': 'patrick',
    
    'paul': 'paul',
    'paulo': 'paul',
    'paolo': 'paul',
    
    'preston': 'preston',
    'pres': 'preston',
    
    'quinn': 'quinn',
    'quin': 'quinn',
    
    'riley': 'riley',
    'ryley': 'riley',
    'rylie': 'riley',
    
    'robert': 'robert',
    'rob': 'robert',
    'bob': 'robert',
    
    'ruben': 'ruben',
    'ruby': 'ruben',
    'rueben': 'ruben',
    
    'russell': 'russell',
    'russ': 'russell',
    'rusty': 'russell',
    
    'ryan': 'ryan',
    'ryann': 'ryan',
    'rian': 'ryan',
    
    'sebastian': 'sebastian',
    'seb': 'sebastian',
    'sebastien': 'sebastian',
    'bash': 'sebastian',
    
    'seth': 'seth',
    'set': 'seth',
    
    'simon': 'simon',
    'si': 'simon',
    'simeon': 'simon',
    
    'spencer': 'spencer',
    'spence': 'spencer',
    'spen': 'spencer',
    
    'thomas': 'thomas',
    'tom': 'thomas',
    'tommy': 'thomas',
    
    'travis': 'travis',
    'trav': 'travis',
    'travvy': 'travis',
    
    'trevor': 'trevor',
    'trev': 'trevor',
    
    'tristan': 'tristan',
    'tris': 'tristan',
    'tristen': 'tristan',
    
    'troy': 'troy',
    'troye': 'troy',
    
    'tyler': 'tyler',
    'ty': 'tyler',
    'tiler': 'tyler',
    
    'vincent': 'vincent',
    'vince': 'vincent',
    'vinny': 'vincent',
    'vinnie': 'vincent',
    
    'wesley': 'wesley',
    'wes': 'wesley',
    'westley': 'wesley',
    
    'weston': 'weston',
    'west': 'weston',
    
    'xavier': 'xavier',
    'xav': 'xavier',
    'zavier': 'xavier',
    
    # Additional female names and variations
    'abigail': 'abigail',
    'abby': 'abigail',
    'abbi': 'abigail',
    'abbey': 'abigail',
    'gail': 'abigail',
    
    'adriana': 'adriana',
    'adrianna': 'adriana',
    'adrienne': 'adriana',
    'adri': 'adriana',
    
    'alexandra': 'alexandra',
    'alex': 'alexandra',
    'alexandria': 'alexandra',
    'sandra': 'alexandra',
    'sandy': 'alexandra',
    'sasha': 'alexandra',
    
    'alexis': 'alexis',
    'alexi': 'alexis',
    'lexi': 'alexis',
    'lexie': 'alexis',
    
    'alicia': 'alicia',
    'alisha': 'alicia',
    'alysha': 'alicia',
    'alice': 'alicia',
    
    'allison': 'allison',
    'alison': 'allison',
    'allyson': 'allison',
    'ally': 'allison',
    'allie': 'allison',
    
    'amanda': 'amanda',
    'mandy': 'amanda',
    'manda': 'amanda',
    'mandie': 'amanda',
    
    'amber': 'amber',
    'amb': 'amber',
    
    'andrea': 'andrea',
    'andria': 'andrea',
    'andie': 'andrea',
    'drea': 'andrea',
    
    'angelica': 'angelica',
    'angelika': 'angelica',
    'angie': 'angelica',
    'angel': 'angelica',
    
    'annabelle': 'annabelle',
    'annabel': 'annabelle',
    'anna': 'annabelle',
    'belle': 'annabelle',
    
    'ariana': 'ariana',
    'ari': 'ariana',
    'arie': 'ariana',
    
    'audrey': 'audrey',
    'audie': 'audrey',
    'aud': 'audrey',
    
    'autumn': 'autumn',
    'autie': 'autumn',
    
    'ava': 'ava',
    'avie': 'ava',
    
    'bella': 'bella',
    'isabella': 'bella',
    'isabel': 'bella',
    'belle': 'bella',
    
    'bianca': 'bianca',
    'bianka': 'bianca',
    
    'brianna': 'brianna',
    'briana': 'brianna',
    'bri': 'brianna',
    'bree': 'brianna',
    
    'brooke': 'brooke',
    'brook': 'brooke',
    
    'caitlin': 'caitlin',
    'caitlyn': 'caitlin',
    'kaitlin': 'caitlin',
    'kaitlyn': 'caitlin',
    'cait': 'caitlin',
    'kate': 'caitlin',
    
    'cassandra': 'cassandra',
    'cassie': 'cassandra',
    'cass': 'cassandra',
    'sandra': 'cassandra',
    'sandy': 'cassandra',
    
    'catherine': 'catherine',
    'katherine': 'catherine',
    'cathy': 'catherine',
    'kathy': 'catherine',
    'cat': 'catherine',
    'kat': 'catherine',
    
    'charlotte': 'charlotte',
    'char': 'charlotte',
    'charlie': 'charlotte',
    'lottie': 'charlotte',
    
    'chelsea': 'chelsea',
    'chelsey': 'chelsea',
    'chelsie': 'chelsea',
    'chels': 'chelsea',
    
    'chloe': 'chloe',
    'clo': 'chloe',
    'chlo': 'chloe',
    
    'christina': 'christina',
    'christine': 'christina',
    'chris': 'christina',
    'tina': 'christina',
    'chrissy': 'christina',
    
    'claire': 'claire',
    'clare': 'claire',
    'clair': 'claire',
    
    'danielle': 'danielle',
    'dani': 'danielle',
    'danni': 'danielle',
    'danny': 'danielle',
    
    'diana': 'diana',
    'diane': 'diana',
    'dianne': 'diana',
    'di': 'diana',
    'didi': 'diana',
    
    'dominique': 'dominique',
    'dom': 'dominique',
    'domi': 'dominique',
    
    'elena': 'elena',
    'helen': 'elena',
    'lena': 'elena',
    
    'elise': 'elise',
    'elisa': 'elise',
    'lise': 'elise',
    
    'elizabeth': 'elizabeth',
    'beth': 'elizabeth',
    'liz': 'elizabeth',
    'lizzie': 'elizabeth',
    'betty': 'elizabeth',
    'betsy': 'elizabeth',
    'bess': 'elizabeth',
    'bessie': 'elizabeth',
    'lisa': 'elizabeth',
    'liza': 'elizabeth',
    'eliza': 'elizabeth',
    'elsie': 'elizabeth',
    'ella': 'elizabeth',
    'ellie': 'elizabeth',
    
    'ella': 'ella',
    'ellie': 'ella',
    'elle': 'ella',
    
    'emily': 'emily',
    'em': 'emily',
    'emmie': 'emily',
    'emma': 'emily',
    'millie': 'emily',
    
    'emma': 'emma',
    'em': 'emma',
    'emmie': 'emma',
    'emmy': 'emma',
    
    'erica': 'erica',
    'erika': 'erica',
    'erika': 'erica',
    'ricka': 'erica',
    
    'erin': 'erin',
    'eryn': 'erin',
    
    'evelyn': 'evelyn',
    'eve': 'evelyn',
    'evie': 'evelyn',
    'lyn': 'evelyn',
    
    'faith': 'faith',
    'fay': 'faith',
    
    'fiona': 'fiona',
    'fi': 'fiona',
    'fifi': 'fiona',
    
    'gabriella': 'gabriella',
    'gabrielle': 'gabriella',
    'gabby': 'gabriella',
    'gab': 'gabriella',
    'brie': 'gabriella',
    
    'grace': 'grace',
    'gracie': 'grace',
    'gray': 'grace',
    
    'hailey': 'hailey',
    'hayley': 'hailey',
    'haleigh': 'hailey',
    'haley': 'hailey',
    'hailie': 'hailey',
    
    'hannah': 'hannah',
    'hanna': 'hannah',
    'han': 'hannah',
    'hannie': 'hannah',
    
    'isabella': 'isabella',
    'isabel': 'isabella',
    'bella': 'isabella',
    'belle': 'isabella',
    'izzy': 'isabella',
    'isa': 'isabella',
    
    'isabelle': 'isabelle',
    'isabel': 'isabelle',
    'bella': 'isabelle',
    'belle': 'isabelle',
    'izzy': 'isabelle',
    
    'jacqueline': 'jacqueline',
    'jackie': 'jacqueline',
    'jacque': 'jacqueline',
    'jacky': 'jacqueline',
    
    'jade': 'jade',
    'jady': 'jade',
    
    'jasmine': 'jasmine',
    'jazz': 'jasmine',
    'jassie': 'jasmine',
    'jasmin': 'jasmine',
    
    'jocelyn': 'jocelyn',
    'joss': 'jocelyn',
    'jossie': 'jocelyn',
    
    'jordan': 'jordan',
    'jordyn': 'jordan',
    'jordy': 'jordan',
    'jordie': 'jordan',
    
    'josephine': 'josephine',
    'josie': 'josephine',
    'jo': 'josephine',
    'joey': 'josephine',
    
    'julia': 'julia',
    'julie': 'julia',
    'juliet': 'julia',
    'jules': 'julia',
    
    'kaitlyn': 'kaitlyn',
    'kaitlin': 'kaitlyn',
    'kait': 'kaitlyn',
    'kate': 'kaitlyn',
    'katie': 'kaitlyn',
    
    'kayla': 'kayla',
    'kay': 'kayla',
    'kaylie': 'kayla',
    
    'kelly': 'kelly',
    'kellie': 'kelly',
    'kel': 'kelly',
    
    'kendall': 'kendall',
    'ken': 'kendall',
    'kenny': 'kendall',
    
    'kennedy': 'kennedy',
    'ken': 'kennedy',
    'kenny': 'kennedy',
    
    'kiana': 'kiana',
    'kianna': 'kiana',
    'ki': 'kiana',
    
    'kylie': 'kylie',
    'ky': 'kylie',
    'kyli': 'kylie',
    
    'lillian': 'lillian',
    'lily': 'lillian',
    'lil': 'lillian',
    'lilly': 'lillian',
    
    'lily': 'lily',
    'lil': 'lily',
    'lilly': 'lily',
    
    'lindsey': 'lindsey',
    'lindsay': 'lindsey',
    'linds': 'lindsey',
    'lindz': 'lindsey',
    
    'madeline': 'madeline',
    'madelyn': 'madeline',
    'maddie': 'madeline',
    'maddy': 'madeline',
    'madie': 'madeline',
    
    'madison': 'madison',
    'maddie': 'madison',
    'maddy': 'madison',
    'madi': 'madison',
    
    'makayla': 'makayla',
    'mikayla': 'makayla',
    'mackenzie': 'makayla',
    'kayla': 'makayla',
    
    'maya': 'maya',
    'maia': 'maya',
    'mya': 'maya',
    
    'megan': 'megan',
    'meghan': 'megan',
    'meg': 'megan',
    'meggie': 'megan',
    
    'melissa': 'melissa',
    'mel': 'melissa',
    'missy': 'melissa',
    'lissa': 'melissa',
    
    'mia': 'mia',
    'miah': 'mia',
    
    'morgan': 'morgan',
    'morgie': 'morgan',
    'morg': 'morgan',
    
    'natalie': 'natalie',
    'nat': 'natalie',
    'nattie': 'natalie',
    'natty': 'natalie',
    
    'naomi': 'naomi',
    'nomi': 'naomi',
    
    'olivia': 'olivia',
    'liv': 'olivia',
    'livvy': 'olivia',
    'ollie': 'olivia',
    
    'paige': 'paige',
    'page': 'paige',
    
    'payton': 'payton',
    'peyton': 'payton',
    'payt': 'payton',
    
    'rachel': 'rachel',
    'rach': 'rachel',
    'rachael': 'rachel',
    'ray': 'rachel',
    
    'rebecca': 'rebecca',
    'becky': 'rebecca',
    'becca': 'rebecca',
    'beckie': 'rebecca',
    'reba': 'rebecca',
    
    'riley': 'riley',
    'ryley': 'riley',
    'rylie': 'riley',
    'ry': 'riley',
    
    'ruby': 'ruby',
    'rue': 'ruby',
    
    'samantha': 'samantha',
    'sam': 'samantha',
    'sammy': 'samantha',
    'sammie': 'samantha',
    
    'savannah': 'savannah',
    'savanna': 'savannah',
    'sav': 'savannah',
    'vannah': 'savannah',
    
    'serena': 'serena',
    'seren': 'serena',
    'sere': 'serena',
    
    'sierra': 'sierra',
    'cierra': 'sierra',
    'sier': 'sierra',
    
    'sophia': 'sophia',
    'sophie': 'sophia',
    'soph': 'sophia',
    
    'sydney': 'sydney',
    'sidney': 'sydney',
    'syd': 'sydney',
    'sid': 'sydney',
    
    'taylor': 'taylor',
    'tay': 'taylor',
    'taytay': 'taylor',
    
    'tiffany': 'tiffany',
    'tiff': 'tiffany',
    'tiffy': 'tiffany',
    
    'valerie': 'valerie',
    'val': 'valerie',
    'valery': 'valerie',
    
    'vanessa': 'vanessa',
    'nessa': 'vanessa',
    'vanny': 'vanessa',
    'van': 'vanessa',
    
    'victoria': 'victoria',
    'vicky': 'victoria',
    'vicki': 'victoria',
    'tori': 'victoria',
    'vic': 'victoria',
    
    'vivian': 'vivian',
    'viv': 'vivian',
    'vivienne': 'vivian',
    'vivi': 'vivian',
    
    'zoe': 'zoe',
    'zoey': 'zoe',
    'zoie': 'zoe',
    'zo': 'zoe',
}

