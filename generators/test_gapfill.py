import pytest

from .GFGenerator import GFQuestions

texts = [
# ('''Government intervention is any action carried out by the government or public entity that affects the market economy with the direct objective of having an impact in the economy, beyond the mere regulation of contracts and provision of public goods. Government intervention advocates defend the use of different economic policies in order to compensate for the flaws of the economic system that give way to large economic imbalances. They believe the Law of Demand and Supply is not sufficient in order to ensure economic equilibriums and government intervention should be used to assure the correct functioning of the economy.  Examples of these economic doctrines include Keynesianism and its branches such as New Keynesian Economics, which rely heavily in fiscal and monetary policies, and Monetarism which have more confidence in monetary policies as they believe fiscal policies will have a negative effect in the long run. On the other hand, there are other economic schools that believe that governments should not have an active role in the economy, and therefore should limit its intervention, as they believe it will have a negative impact in the economy. They believe that the economy should be left to run in a laissez-faire way and it will find its optimal equilibrium.  Advocates of limited intervention include liberalism, the Austrian school and New Classical Macroeconomics.''',
	# "Government Intervention",
	# "Economics"
	# ),
('''Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to effectively perform a specific task without using explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as ‘training data’, in order to make predictions or decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in a wide variety of applications, such as email filtering, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory data analysis through unsupervised learning. In its application across business problems, machine learning is also referred to as predictive analytics.''',
	"Machine Learning",
	"Computer Science"),
# ('''In programming languages, a type system is a set of rules that assigns a property called type to the various constructs of a computer program, such as variables, expressions, functions or modules. These types formalize and enforce the otherwise implicit categories the programmer uses for algebraic data types, data structures, or other components (e.g. "string", "array of float", "function returning boolean"). The main purpose of a type system is to reduce possibilities for bugs in computer programs by defining interfaces between different parts of a computer program, and then checking that the parts have been connected in a consistent way. This checking can happen statically (at compile time), dynamically (at run time), or as a combination of static and dynamic checking. Type systems have other purposes as well, such as expressing business rules, enabling certain compiler optimizations, allowing for multiple dispatch, providing a form of documentation, etc.  A type system associates a type with each computed value and, by examining the flow of these values, attempts to ensure or prove that no type errors can occur. The given type system in question determines exactly what constitutes a type error, but in general the aim is to prevent operations expecting a certain kind of value from being used with values for which that operation does not make sense (logic errors). Type systems are often specified as part of programming languages, and built into the interpreters and compilers for them; although the type system of a language can be extended by optional tools that perform added kinds of checks using the language's original type syntax and grammar. ''',
	# "Type System",
	# "Computer Science"),
('''An atom is the smallest constituent unit of ordinary matter that has the properties of a chemical element. Every solid, liquid, gas, and plasma is composed of neutral or ionized atoms. Atoms are extremely small; typical sizes are around 100 picometers (a ten-billionth of a meter, in the short scale). Atoms are small enough that attempting to predict their behavior using classical physics – as if they were billiard balls, for example – gives noticeably incorrect predictions due to quantum effects. Through the development of physics, atomic models have incorporated quantum principles to better explain and predict this behavior. Every atom is composed of a nucleus and one or more electrons bound to the nucleus. The nucleus is made of one or more protons and typically a similar number of neutrons. Protons and neutrons are called nucleons. More than 99.94% of an atom's mass is in the nucleus. The protons have a positive electric charge, the electrons have a negative electric charge, and the neutrons have no electric charge. If the number of protons and electrons are equal, that atom is electrically neutral. If an atom has more or fewer electrons than protons, then it has an overall negative or positive charge, respectively, and it is called an ion.''',
	"The Atom",
	"Physics"),
# ('''The Suez Crisis, or the Second Arab–Israeli War, also named the Tripartite Aggression in the Arab world and Operation Kadesh or Sinai War in Israel, was an invasion of Egypt in late 1956 by Israel, followed by the United Kingdom and France. The aims were to regain Western control of the Suez Canal and to remove Egyptian President Gamal Abdel Nasser, who had just nationalized the canal. After the fighting had started, political pressure from the United States, the Soviet Union and the United Nations led to a withdrawal by the three invaders. The episode humiliated the United Kingdom and France and strengthened Nasser.  On 29 October, Israel invaded the Egyptian Sinai. Britain and France issued a joint ultimatum to cease fire, which was ignored. On 5 November, Britain and France landed paratroopers along the Suez Canal. The Egyptian forces were defeated, but they did block the canal to all shipping. It later became clear that the Israeli invasion and the subsequent Anglo-French attack had been planned beforehand by the three countries.  The three allies had attained a number of their military objectives, but the canal was useless. Heavy political pressure from the United States and the USSR led to a withdrawal. U.S. President Dwight D. Eisenhower had strongly warned Britain not to invade; he threatened serious damage to the British financial system by selling the US government's pound sterling bonds. Historians conclude the crisis "signified the end of Great Britain's role as one of the world's major powers". The Suez Canal was closed from October 1956 until March 1957. Israel fulfilled some of its objectives, such as attaining freedom of navigation through the Straits of Tiran, which Egypt had blocked to Israeli shipping since 1950. As a result of the conflict, the United Nations created the UNEF Peacekeepers to police the Egyptian–Israeli border, British Prime Minister Anthony Eden resigned, Canadian Minister of External Affairs Lester Pearson won the Nobel Peace Prize, and the USSR may have been emboldened to invade Hungary.''',
# 	"The Suez Crisis",
# 	"History"),
# ('''Kepler’s laws of planetary motion, in astronomy and classical physics, laws describing the motions of the planets in the solar system. They were derived by the German astronomer Johannes Kepler, whose analysis of the observations of the 16th-century Danish astronomer Tycho Brahe enabled him to announce his first two laws in the year 1609 and a third law nearly a decade later, in 1618. Kepler himself never numbered these laws or specially distinguished them from his other discoveries. Kepler’s three laws of planetary motion can be stated as follows: (1) All planets move about the Sun in elliptical orbits, having the Sun as one of the foci. (2) A radius vector joining any planet to the Sun sweeps out equal areas in equal lengths of time. (3) The squares of the sidereal periods (of revolution) of the planets are directly proportional to the cubes of their mean distances from the Sun. Knowledge of these laws, especially the second (the law of areas), proved crucial to Sir Isaac Newton in 1684–85, when he formulated his famous law of gravitation between Earth and the Moon and between the Sun and the planets, postulated by him to have validity for all objects anywhere in the universe. Newton showed that the motion of bodies subject to central gravitational force need not always follow the elliptical orbits specified by the first law of Kepler but can take paths defined by other, open conic curves; the motion can be in parabolic or hyperbolic orbits, depending on the total energy of the body. Thus, an object of sufficient energy—e.g., a comet—can enter the solar system and leave again without returning. From Kepler’s second law, it may be observed further that the angular momentum of any planet about an axis through the Sun and perpendicular to the orbital plane is also unchanging.''',
# 	"Kepler's Laws",
# 	"Physics"),
('''In computer science, a queue is a collection in which the entities in the collection are kept in order and the principal (or only) operations on the collection are the addition of entities to the rear terminal position, known as enqueue, and removal of entities from the front terminal position, known as dequeue. This makes the queue a First-In-First-Out (FIFO) data structure. In a FIFO data structure, the first element added to the queue will be the first one to be removed. This is equivalent to the requirement that once a new element is added, all elements that were added before have to be removed before the new element can be removed. Often a peek or front operation is also entered, returning the value of the front element without dequeuing it. A queue is an example of a linear data structure, or more abstractly a sequential collection.''',
	"Queue",
	"Computer Science"),
# ("""The Cold War was a period of geopolitical tension between the Soviet Union with its satellite states (the Eastern Bloc), and the United States with its allies (the Western Bloc) after World War II. A common historiography of the conflict begins between 1946, the year U.S. diplomat George F. Kennan's "Long Telegram" from Moscow cemented a U.S. foreign policy of containment of Soviet expansionism threatening strategically vital regions, and the Truman Doctrine of 1947, and ending between the Revolutions of 1989, which ended communism in Eastern Europe as well as in other areas, and the 1991 collapse of the USSR, when nations of the Soviet Union abolished communism and restored their independence. The term "cold" is used because there was no large-scale fighting directly between the two sides, but they each supported major regional conflicts known as proxy wars. The conflict split the temporary wartime alliance against Nazi Germany and its allies, leaving the USSR and the US as two superpowers with profound economic and political differences.  The capitalist West was led by the United States, a federal republic with a two-party presidential system, as well as the other First World nations of the Western Bloc that were generally liberal democratic with a free press and independent organizations, but were economically and politically entwined with a network of banana republics and other authoritarian regimes, most of which were the Western Bloc's former colonies. Some major Cold War frontlines such as Indochina, Indonesia, and the Congo were still Western colonies in 1947. The Soviet Union, on the other hand, was a self-proclaimed Marxist–Leninist state that imposed a totalitarian regime that was led by a small committee, the Politburo. The Party had full control of the state, the press, the military, the economy, and local organizations throughout the Second World, including the Warsaw Pact and other satellites. The Kremlin funded communist parties around the world but was challenged for control by Mao's China following the Sino-Soviet split of the 1960s. As nearly all the colonial states achieved independence 1945-1960, they became Third World battlefields in the Cold War.  India, Indonesia, and Yugoslavia took the lead in promoting neutrality with the Non-Aligned Movement, but it never had much power in its own right. The Soviet Union and the United States never engaged directly in full-scale armed combat. However, both were heavily armed in preparation for a possible all-out nuclear world war. China and the United States fought an undeclared high-casualty war in Korea 1950-53 that resulted in a stalemate. Each side had a nuclear strategy that discouraged an attack by the other side, on the basis that such an attack would lead to the total destruction of the attacker—the doctrine of mutually assured destruction (MAD). Aside from the development of the two sides' nuclear arsenals, and their deployment of conventional military forces, the struggle for dominance was expressed via proxy wars around the globe, psychological warfare, massive propaganda campaigns and espionage, far-reaching embargoes, rivalry at sports events, and technological competitions such as the Space Race.  The first phase of the Cold War began in the first two years after the end of the Second World War in 1945. The USSR consolidated its control over the states of the Eastern Bloc, while the United States began a strategy of global containment to challenge Soviet power, extending military and financial aid to the countries of Western Europe (for example, supporting the anti-communist side in the Greek Civil War) and creating the NATO alliance. The Berlin Blockade (1948–49) was the first major crisis of the Cold War. With the victory of the Communist side in the Chinese Civil War and the outbreak of the Korean War (1950–1953), the conflict expanded. The USSR and the US competed for influence in Latin America and the decolonizing states of Africa and Asia. The Soviets suppressed the Hungarian Revolution of 1956. The expansion and escalation sparked more crises, such as the Suez Crisis (1956), the Berlin Crisis of 1961, and the Cuban Missile Crisis of 1962, which was perhaps the closest the two sides came to nuclear war. Meanwhile, an international peace movement took root and grew among citizens around the world, first in Japan from 1954, when people became concerned about nuclear weapons testing, but soon also in Europe and the US. The peace movement, and in particular the anti-nuclear movement, gained pace and popularity from the late 1950s and early 1960s, and continued to grow through the '70s and '80s with large protest marches, demonstrations, and various non-parliamentary activism opposing war and calling for global nuclear disarmament. Following the Cuban Missile Crisis, a new phase began that saw the Sino-Soviet split complicate relations within the Communist sphere, while US allies, particularly France, demonstrated greater independence of action. The USSR crushed the 1968 Prague Spring liberalization program in Czechoslovakia, while the US experienced internal turmoil from the civil rights movement and opposition to the Vietnam War (1955–75), which ended with the defeat of the US-backed Republic of Vietnam, prompting further adjustments.  By the 1970s, both sides had become interested in making allowances in order to create a more stable and predictable international system, ushering in a period of détente that saw Strategic Arms Limitation Talks and the US opening relations with the People's Republic of China as a strategic counterweight to the Soviet Union. Détente collapsed at the end of the decade with the beginning of the Soviet–Afghan War in 1979. The early 1980s were another period of elevated tension, with the Soviet downing of KAL Flight 007 and the "Able Archer" NATO military exercises, both in 1983. The United States increased diplomatic, military, and economic pressures on the Soviet Union, at a time when the communist state was already suffering from economic stagnation. On 12 June 1982, a million protesters gathered in Central Park, New York to call for an end to the Cold War arms race and nuclear weapons in particular. In the mid-1980s, the new Soviet leader Mikhail Gorbachev introduced the liberalizing reforms of perestroika ("reorganization", 1987) and glasnost ("openness", c. 1985) and ended Soviet involvement in Afghanistan. Pressures for national independence grew stronger in Eastern Europe, especially Poland. Gorbachev meanwhile refused to use Soviet troops to bolster the faltering Warsaw Pact regimes as had occurred in the past. The result in 1989 was a wave of revolutions that peacefully (with the exception of the Romanian Revolution) overthrew all of the communist regimes of Central and Eastern Europe. The Communist Party of the Soviet Union itself lost control and was banned following an abortive coup attempt in August 1991. This in turn led to the formal dissolution of the USSR in December 1991 and the collapse of communist regimes in other countries such as Mongolia, Cambodia, and South Yemen. The United States remained as the world's only superpower.  The Cold War and its events have left a significant legacy. It is often referred to in popular culture, especially in media featuring themes of espionage (notably the internationally successful James Bond book and film franchise) and the threat of nuclear warfare. Meanwhile, a renewed state of tension between the Soviet Union's successor state, Russia, and the United States in the 2010s (including its Western allies) and growing tension between an increasingly powerful China and the U.S. and its Western Allies have each been referred to as the Second Cold War.""", 'The Cold War', 'History'),
# ("""Privatization (also spelled privatisation) can mean different things including moving something from the public sector into the private sector. It is also sometimes used as a synonym for deregulation when a heavily regulated private company or industry becomes less regulated. Government functions and services may also be privatized; in this case, private entities are tasked with the implementation of government programs or performance of government services that had previously been the purview of state-run agencies. Some examples include revenue collection, law enforcement, and prison management.Another definition is the purchase of all outstanding shares of a publicly traded company by private investors, or the sale of a state-owned enterprise or municipally owned corporation to private investors. In the case of a for-profit company, the shares are then no longer traded at a stock exchange, as the company became private through private equity; in the case the partial or full sale of a state-owned enterprise or municipally owned corporation to private owners shares may be traded in the public market for the first time, or for the first time since an enterprise's previous nationalization. The second such type of privatization is the demutualization of a mutual organization, cooperative, or public-private partnership in order to form a joint-stock company.""", 'Privatization', 'Economics'),
# ("""Muscle contraction is the activation of tension-generating sites within muscle fibers. In physiology, muscle contraction does not necessarily mean muscle shortening because muscle tension can be produced without changes in muscle length such as holding a heavy book or a dumbbell at the same position. The termination of muscle contraction is followed by muscle relaxation, which is a return of the muscle fibers to their low tension-generating state.  Muscle contractions can be described based on two variables: length and tension. A muscle contraction is described as isometric if the muscle tension changes but the muscle length remains the same. In contrast, a muscle contraction is isotonic if muscle tension remains the same throughout the contraction. If the muscle length shortens, the contraction is concentric; if the muscle length lengthens, the contraction is eccentric. In natural movements that underlie locomotor activity, muscle contractions are multifaceted as they are able to produce changes in length and tension in a time-varying manner. Therefore, neither length nor tension is likely to remain the same in muscles that contract during locomotor activity.  In vertebrates, skeletal muscle contractions are neurogenic as they require synaptic input from motor neurons to produce muscle contractions. A single motor neuron is able to innervate multiple muscle fibers, thereby causing the fibers to contract at the same time. Once innervated, the protein filaments within each skeletal muscle fiber slide past each other to produce a contraction, which is explained by the sliding filament theory. The contraction produced can be described as a twitch, summation, or tetanus, depending on the frequency of action potentials. In skeletal muscles, muscle tension is at its greatest when the muscle is stretched to an intermediate length as described by the length-tension relationship.  Unlike skeletal muscle, the contractions of smooth and cardiac muscles are myogenic (meaning that they are initiated by the smooth or heart muscle cells themselves instead of being stimulated by an outside event such as nerve stimulation), although they can be modulated by stimuli from the autonomic nervous system. The mechanisms of contraction in these muscle tissues are similar to those in skeletal muscle tissues.""", 'Muscle Contraction', 'Biology'),
("""Vegetable oils and animal fats are the traditional materials that are saponified. These greasy materials, triesters called triglycerides, are mixtures derived from diverse fatty acids. Triglycerides can be converted to soap in either a one- or a two-step process. In the traditional one-step process, the triglyceride is treated with a strong base (e.g. lye), which cleaves to the ester bond, releasing fatty acid salts (soaps) and glycerol. This process is also the main industrial method for producing glycerol. In some soap-making, the glycerol is left in the soap. If necessary, soaps may be precipitated by salting it out with sodium chloride.  Fat in a corpse converts into adipocere, often called "grave wax". This process is more common where the amount of fatty tissue is high and the agents of decomposition are absent or only minutely present.  The saponification value is the amount of base required to saponify a fat sample. Soap makers formulate their recipes with a small deficit of lye to account for the unknown deviation of saponification value between their oil batch and laboratory averages.""", 'Saponification', 'Chemistry'),
# ("""Kublai (/ˈkuːblaɪ/; Mongolian: Хубилай, romanized: Hubilai; Chinese: 忽必烈; pinyin: Hūbìliè) was the fifth Khagan (Great Khan) of the Mongol Empire (Ikh Mongol Uls), reigning from 1260 to 1294 (although due to the division of the empire this was a nominal position). He also founded the Yuan dynasty in China as a conquest dynasty in 1271, and ruled as the first Yuan emperor until his death in 1294. Kublai was the fourth son of Tolui (his second son with Sorghaghtani Beki) and a grandson of Genghis Khan. He succeeded his older brother Möngke as Khagan in 1260, but had to defeat his younger brother Ariq Böke in the Toluid Civil War lasting until 1264. This episode marked the beginning of disunity in the empire. Kublai's real power was limited to China and Mongolia, though as Khagan he still had influence in the Ilkhanate and, to a significantly lesser degree, in the Golden Horde. If one counts the Mongol Empire at that time as a whole, his realm reached from the Pacific Ocean to the Black Sea, from Siberia to what is now Afghanistan. In 1271, Kublai established the Yuan dynasty, which ruled over present-day Mongolia, China, Korea, and some adjacent areas, and assumed the role of Emperor of China. By 1279, the Mongol conquest of the Song dynasty was completed and Kublai became the first non-Han emperor to conquer all of China. The imperial portrait of Kublai was part of an album of the portraits of Yuan emperors and empresses, now in the collection of the National Palace Museum in Taipei. White, the color of the royal costume of Kublai, was the imperial color of the Yuan dynasty.""", 'Kublai Khan', "History"),
# ("""Genghis Khan (born Temüjin, c. 1162 – August 18, 1227) was the founder and first Great Khan of the Mongol Empire, which became the largest contiguous empire in history after his death. He came to power by uniting many of the nomadic tribes of Northeast Asia. After founding the Empire and being proclaimed Genghis Khan, he launched the Mongol invasions that conquered most of Eurasia. Campaigns initiated in his lifetime include those against the Qara Khitai, Caucasus, and Khwarazmian, Western Xia and Jin dynasties. These campaigns were often accompanied by large-scale massacres of the civilian populations – especially in the Khwarazmian and Western Xia controlled lands. By the end of his life, the Mongol Empire occupied a substantial portion of Central Asia and China. Before Genghis Khan died he assigned Ögedei Khan as his successor. Later his grandsons split his empire into khanates. Genghis Khan died in 1227 after defeating the Western Xia. By his request, his body was buried in an unmarked grave somewhere in Mongolia. His descendants extended the Mongol Empire across most of Eurasia by conquering or creating vassal states in all of modern-day China, Korea, the Caucasus, Central Asia, and substantial portions of Eastern Europe and Southwest Asia. Many of these invasions repeated the earlier large-scale slaughters of local populations. As a result, Genghis Khan and his empire have a fearsome reputation in local histories. Beyond his military accomplishments, Genghis Khan also advanced the Mongol Empire in other ways. He decreed the adoption of the Uyghur script as the Mongol Empire's writing system. He also practised meritocracy and encouraged religious tolerance in the Mongol Empire, and unified the nomadic tribes of Northeast Asia. Present-day Mongolians regard him as the founding father of Mongolia. Genghis Khan was known for the brutality of his campaigns, and is considered by many to have been a genocidal ruler. However, he is also credited with bringing the Silk Road under one cohesive political environment. This brought relatively easy communication and trade between Northeast Asia, Muslim Southwest Asia, and Christian Europe, expanding the cultural horizons of all three areas.""", 'Genghis Khan', 'History'),
("""The first use of rubber was by the indigenous cultures of Mesoamerica. The earliest archeological evidence of the use of natural latex from the Hevea tree comes from the Olmec culture, in which rubber was first used for making balls for the Mesoamerican ballgame. Rubber was later used by the Maya and Aztec cultures – in addition to making balls Aztecs used rubber for other purposes such as making containers and to make textiles waterproof by impregnating them with the latex sap. The Pará rubber tree is indigenous to South America. Charles Marie de La Condamine is credited with introducing samples of rubber to the Académie Royale des Sciences of France in 1736. In 1751, he presented a paper by François Fresneau to the Académie (published in 1755) that described many of rubber's properties. This has been referred to as the first scientific paper on rubber. In England, Joseph Priestley, in 1770, observed that a piece of the material was extremely good for rubbing off pencil marks on paper, hence the name "rubber". It slowly made its way around England. In 1764 François Fresnau discovered that turpentine was a rubber solvent. Giovanni Fabbroni is credited with the discovery of naphtha as a rubber solvent in 1779. South America remained the main source of latex rubber used during much of the 19th century. The rubber trade was heavily controlled by business interests but no laws expressly prohibited the export of seeds or plants. In 1876, Henry Wickham smuggled 70,000 Pará rubber tree seeds from Brazil and delivered them to Kew Gardens, England. Only 2,400 of these germinated. Seedlings were then sent to India, British Ceylon (Sri Lanka), Dutch East Indies (Indonesia), Singapore, and British Malaya. Malaya (now Peninsular Malaysia) was later to become the biggest producer of rubber.[10] In the early 1900s, the Congo Free State in Africa was also a significant source of natural rubber latex, mostly gathered by forced labor. King Leopold II's colonial state brutally enforced production quotas. Tactics to enforce the rubber quotas included removing the hands of victims to prove they had been killed. Soldiers often came back from raids with baskets full of chopped-off hands. Villages that resisted were razed to encourage better compliance locally. See Atrocities in the Congo Free State for more information on the rubber trade in the Congo Free State in the late 1800s and early 1900s. Liberia and Nigeria started production.""", 'Rubber', 'Science')
]

@pytest.fixture
def instance():
	g = GFQuestions(testing=True, model="en_core_web_lg")
	yield g

def test_keywords():
	for text, title, subject in texts:
		response = GFQuestions.generate_keywords(text)
		print(response)
		assert response != None

def test_questions(instance):
	for text, title, subject in texts:
		d = {
		"text": text,
		"title": title,
		"subject": subject
		}
		response = instance.generate_questions(d)
		print(response)
		assert response != None
