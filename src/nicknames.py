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
}

